# examples/hull_benchmark.py
#2.89 Sharpe Config:
#python examples/hull_benchmark.py  --data datasets/train.csv --era-size 180 --holdout-eras 1 --to-signal-mult 1600 --lastn 180  --impute ffill --target-col forward_excess
from __future__ import annotations
import sys, time, argparse, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

# allow "pip -e ." style local import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------- Optional baselines (skipped if not installed) --------
def _try_import_xgb():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except Exception:
        return None

def _try_import_lgbm():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    except Exception:
        return None

def _try_import_cat():
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor
    except Exception:
        return None

# -------- PackBoost (your lib) --------
try:
    from packboost.booster import PackBoost
    from packboost.config import PackBoostConfig
except Exception as e:
    raise RuntimeError("PackBoost not found on PYTHONPATH. Install with `pip -e .` at repo root.") from e


@dataclass
class BenchmarkResult:
    name: str
    fit_time: float
    predict_time: float
    r2: float
    era_mean_corr: float
    era_sharpe: float
    lastn_sharpe: float
    hull_penalized: float | None  # None if required columns missing


# ---------- Metrics ----------
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return np.nan
    sa, sb = a.std(), b.std()
    if not np.isfinite(sa) or not np.isfinite(sb) or sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def era_metrics(era: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> Tuple[float, float]:
    df = pd.DataFrame({"era": era, "y": y, "p": yhat})
    cors = df.groupby("era", sort=False).apply(
        lambda g: _safe_corr(g["y"].values, g["p"].values)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(cors) == 0:
        return float("nan"), float("nan")
    mean = float(cors.mean())
    std = float(cors.std(ddof=0)) if len(cors) > 1 else np.nan
    sharpe = float(mean / std) if (std is not None and std > 0) else np.nan
    return mean, sharpe

def annualized_sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    if returns.size < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float((np.sqrt(trading_days) * mu) / sd)


# ---------- Column inference & feature groups ----------
GROUP_PREFIXES = ("M", "E", "I", "P", "V", "S", "MOM", "D")

def infer_columns(df: pd.DataFrame, era_col_arg: Optional[str],
                  target_col_arg: Optional[str], include_groups: Optional[List[str]]
                  ) -> Tuple[List[str], str, str]:
    # Era
    era_col = era_col_arg or ("date_id" if "date_id" in df.columns else None)
    if era_col is None:
        raise ValueError("Couldn't find an era column. Pass --era-col or include 'date_id' in CSV.")

    # Target: prefer excess returns
    if target_col_arg and target_col_arg in df.columns:
        target_col = target_col_arg
    else:
        if "market_forward_excess_returns" in df.columns:
            target_col = "market_forward_excess_returns"
        elif "forward_returns" in df.columns:
            target_col = "forward_returns"  # convert to excess later if rf exists
            warnings.warn(
                "Using forward_returns as target. Will convert to excess if risk_free_rate is present; "
                "otherwise results may misalign with competition target."
            )
        else:
            raise ValueError("Expected 'market_forward_excess_returns' (preferred) or 'forward_returns'.")

    # Features: numeric columns excluding meta; optionally filter by group prefixes
    meta = {era_col, "risk_free_rate", "market_forward_excess_returns", "forward_returns",
            "lagged_forward_returns", "lagged_risk_free_rate", "lagged_market_forward_excess_returns",
            "is_scored"}
    num_cols = [c for c, dt in df.dtypes.items() if np.issubdtype(dt, np.number)]
    feats = [c for c in num_cols if c not in meta]

    if include_groups:
        allow = tuple(g.strip().upper() for g in include_groups)
        def in_group(col: str) -> bool:
            up = col.upper()
            return any(up.startswith(g + "_") or up.startswith(g) for g in allow)
        feats = [c for c in feats if in_group(c)]

    if not feats:
        raise ValueError("No numeric feature columns found after filtering.")
    return feats, target_col, era_col


# ---------- Era bucketing ----------
def bucketize_era(era_series: pd.Series, size: int) -> pd.Series:
    """Group consecutive date_id values into fixed-size buckets (size=21 ~ monthly-ish)."""
    if size <= 1:
        return era_series.astype(np.int32)
    base = int(era_series.min())
    return (((era_series.astype(np.int64) - base) // int(size))).astype(np.int32)


# ---------- Feature engineering (lag1, z21, mom63 on selected groups) ----------
def engineer_features(df: pd.DataFrame, feats: List[str], date_col: str,
                      groups_for_fx: List[str], window_z: int = 21, window_mom: int = 63,
                      cap_cols: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds:
      - f_lag1
      - f_z{window_z} = (f - rolling_mean)/rolling_std
      - f_mom{window_mom} = f - rolling_mean_{window_mom}
    Only for columns whose prefix is in groups_for_fx.
    """
    df = df.sort_values(date_col).copy()
    new_cols: List[str] = []
    pref_set = tuple(g.upper() for g in groups_for_fx)

    cols = feats if cap_cols is None else feats[:cap_cols]

    for c in cols:
        cu = c.upper()
        if not any(cu.startswith(p) for p in pref_set):
            continue
        s = df[c].astype(np.float32)
        df[c + f"_lag1"] = s.shift(1)
        new_cols.append(c + f"_lag1")

        m = s.rolling(window_z, min_periods=5).mean()
        sd = s.rolling(window_z, min_periods=5).std(ddof=0)
        z = (s - m) / sd.replace(0, np.nan)
        df[c + f"_z{window_z}"] = z.astype(np.float32)
        new_cols.append(c + f"_z{window_z}")

        m63 = s.rolling(window_mom, min_periods=10).mean()
        mom = s - m63
        df[c + f"_mom{window_mom}"] = mom.astype(np.float32)
        new_cols.append(c + f"_mom{window_mom}")

    return df, feats + new_cols


# ---------- Directional agreement (train-only, NaN-robust) ----------
def dir_agreement_filter(X_tr: np.ndarray, y_tr: np.ndarray, feat_names: List[str], thresh: float) -> List[int]:
    """
    Keep features whose sign agreement with y exceeds thresh.
    Agreement = mean( sign(zscore(x)) == sign(y) ) over non-zero pairs.
    """
    mu  = np.nanmean(X_tr, axis=0)
    sig = np.nanstd (X_tr, axis=0) + 1e-12
    Xz  = (X_tr - mu) / sig
    ysig = np.sign(y_tr)
    keep = []
    for j in range(Xz.shape[1]):
        xs = np.sign(Xz[:, j])
        mask = (~np.isnan(xs)) & (ysig != 0)
        if not np.any(mask):
            continue
        agree = float(np.mean(xs[mask] == ysig[mask]))
        if agree >= thresh:
            keep.append(j)
    if not keep:
        # fall back to top-50 by abs Pearson corr if filter is too strict
        zsafe = np.nan_to_num(Xz, copy=False)
        corr_row = np.corrcoef(zsafe.T, y_tr)[-1, :-1]
        corr_row = np.nan_to_num(corr_row, copy=False)
        keep = list(np.argsort(-np.abs(corr_row))[:min(50, Xz.shape[1])])
    return keep


# ---------- Splits (on bucketed eras) ----------
def temporal_holdout_masks(era_series_bucketed: pd.Series, holdout_eras: Optional[int], test_frac: float = 0.2):
    eras_sorted = np.array(sorted(era_series_bucketed.unique()))
    if holdout_eras is not None and holdout_eras > 0:
        k = min(holdout_eras, max(1, len(eras_sorted) - 1))
    else:
        k = max(1, int(round(test_frac * len(eras_sorted))))
    test_eras = set(eras_sorted[-k:])
    tr = ~era_series_bucketed.isin(test_eras)
    te =  era_series_bucketed.isin(test_eras)
    return tr.values, te.values, eras_sorted[-k:]

def era_time_series_splits(era_series_bucketed: pd.Series, n_splits=5, embargo_eras=0):
    eras = np.array(sorted(era_series_bucketed.unique()))
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr_e_idx, te_e_idx in tss.split(eras):
        tr_eras = eras[tr_e_idx]
        te_eras = eras[te_e_idx]
        if embargo_eras > 0 and tr_eras.size > embargo_eras:
            tr_eras = tr_eras[:-embargo_eras]
        tr_mask = era_series_bucketed.isin(tr_eras).values
        te_mask = era_series_bucketed.isin(te_eras).values
        yield tr_mask, te_mask


# ---------- Last-180 mask by global date_id (no leakage) ----------
def lastn_mask_by_date(df: pd.DataFrame, date_col: str, n: int = 180) -> np.ndarray:
    last_ids = sorted(df[date_col].unique())[-n:]
    return df[date_col].isin(last_ids).values


# ---------- Hull penalized-Sharpe (integrated) ----------
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0
TRADING_DAYS_PER_YEAR = 252

def hull_penalized_sharpe(signal: np.ndarray,
                          forward_returns: np.ndarray,
                          risk_free_rate: np.ndarray) -> float:
    """Competition metric: volatility-adjusted Sharpe with penalties."""
    sig = np.asarray(signal, dtype=np.float64)
    fr  = np.asarray(forward_returns, dtype=np.float64)
    rf  = np.asarray(risk_free_rate, dtype=np.float64)

    if sig.shape != fr.shape or sig.shape != rf.shape:
        raise ValueError("Inputs must have the same shape.")
    if not np.isfinite(sig).all() or not np.isfinite(fr).all() or not np.isfinite(rf).all():
        raise ValueError("Inputs contain NaN/inf.")
    if sig.max() > MAX_INVESTMENT + 1e-12:
        raise ValueError(f"Position of {sig.max():.6g} exceeds maximum of {MAX_INVESTMENT}.")
    if sig.min() < MIN_INVESTMENT - 1e-12:
        raise ValueError(f"Position of {sig.min():.6g} below minimum of {MIN_INVESTMENT}.")

    # Strategy returns as convex mix of cash (rf) and market (fr)
    strategy_returns = rf * (1.0 - sig) + sig * fr
    strategy_excess  = strategy_returns - rf

    # Geometric mean style Sharpe
    strategy_excess_cum = np.prod(1.0 + strategy_excess)
    strategy_mean_excess = strategy_excess_cum ** (1.0 / len(sig)) - 1.0
    strategy_std = np.std(strategy_returns, ddof=1)
    if strategy_std == 0.0 or not np.isfinite(strategy_std):
        return 0.0
    sharpe = (strategy_mean_excess / strategy_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

    strategy_vol = float(strategy_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)

    # Market stats
    market_excess = fr - rf
    market_excess_cum = np.prod(1.0 + market_excess)
    market_mean_excess = market_excess_cum ** (1.0 / len(sig)) - 1.0
    market_std = np.std(fr, ddof=1)
    market_vol = float(market_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    if market_vol == 0.0 or not np.isfinite(market_vol):
        return 0.0

    # Penalties
    excess_vol = max(0.0, strategy_vol / market_vol - 1.2) if market_vol > 0.0 else 0.0
    vol_penalty = 1.0 + excess_vol

    return_gap = max(0.0, (market_mean_excess - strategy_mean_excess) * 100.0 * TRADING_DAYS_PER_YEAR)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return float(min(adjusted_sharpe, 1_000_000.0))


# ---------- Exclusive Feature Bundling (PackBoost path only) ----------
class EFBBundler:
    """
    Greedy EFB that bundles (near) mutually-exclusive features: by default only 'D*' columns.
    - Fit on TRAIN rows only (to avoid leakage).
    - Transform concatenates: [non-bundled features] + [sum over each bundle].
    - If any bundle sees overlap at inference, values are summed (rare if trained with strict conflicts=0).
    """
    def __init__(self, bundles: List[List[int]], not_sel: List[int], bundle_names: List[str]):
        self.bundles = bundles
        self.not_sel = not_sel
        self.bundle_names = bundle_names

    @staticmethod
    def _select_prefix_idx(feat_names: List[str], prefixes: List[str]) -> List[int]:
        prefs = tuple(p.upper() for p in prefixes)
        idx = []
        for i, n in enumerate(feat_names):
            nu = n.upper()
            if nu.startswith(prefs) or any(nu.startswith(p + "_") for p in prefs):
                idx.append(i)
        return idx

    @classmethod
    def fit(cls, X_tr: np.ndarray, feat_names: List[str],
            prefixes: List[str], max_conflicts: int = 0, max_bundle_size: int = 8) -> Optional["EFBBundler"]:
        sel = cls._select_prefix_idx(feat_names, prefixes)
        if len(sel) <= 1:
            return None

        # boolean nonzero mask on selected features
        M = (~np.isclose(np.nan_to_num(X_tr[:, sel], copy=False), 0.0)).astype(np.int8)
        col_nnz = M.sum(axis=0)
        if col_nnz.sum() == 0:
            return None

        # conflicts counts on selected set
        # conflicts[i,j] = #rows where both i and j are nonzero
        conflicts = (M.T @ M).astype(np.int32)

        # Greedy: add features (sparsest first) to existing bundles if no conflicts beyond threshold
        order = np.argsort(col_nnz)  # sparse -> dense
        bundles_sel_idx: List[List[int]] = []
        for k in order:
            placed = False
            for B in bundles_sel_idx:
                if len(B) >= max_bundle_size:
                    continue
                # check conflicts with all in current bundle
                if all(conflicts[k, b] <= max_conflicts for b in B):
                    B.append(k)
                    placed = True
                    break
            if not placed:
                bundles_sel_idx.append([k])

        # Map bundles from local sel-index to original feature indices
        bundles_orig = [[sel[k] for k in B] for B in bundles_sel_idx]
        not_sel = [i for i in range(len(feat_names)) if i not in sel]
        bnames = [f"EFB_b{bi}:[{','.join(feat_names[j] for j in B)}]" for bi, B in enumerate(bundles_orig)]
        return EFBBundler(bundles_orig, not_sel, bnames)

    def transform(self, X: np.ndarray) -> np.ndarray:
        parts = [X[:, self.not_sel]]
        for B in self.bundles:
            # sum since (near) mutually exclusive; nan-safe
            parts.append(np.nansum(X[:, B], axis=1, dtype=np.float32, keepdims=True))
        return np.concatenate(parts, axis=1)

    def new_feature_names(self, feat_names: List[str]) -> List[str]:
        return [feat_names[i] for i in self.not_sel] + self.bundle_names


# ---------- PackBoost helpers ----------
def make_packboost(args, seed: int):
    cfg = PackBoostConfig(
        pack_size=args.pack_size,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        lambda_l2=args.lambda_l2,
        lambda_dro=args.lambda_dro,
        min_samples_leaf=args.min_samples_leaf,
        max_bins=args.max_bins,
        k_cuts=args.k_cuts,
        random_state=seed,
        layer_feature_fraction=args.layer_feature_fraction,
        direction_weight=args.direction_weight,
        prebinned=bool(args.prebinned),
    )
    if (args.n_trees % cfg.pack_size) != 0:
        raise ValueError("n_trees must be divisible by pack_size.")
    rounds = args.n_trees // cfg.pack_size
    booster = PackBoost(cfg)
    return booster, rounds


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Hull benchmark: PackBoost vs baselines with era-aware eval.")
    ap.add_argument("--data", type=Path, default=Path("datasets/train.csv"))
    ap.add_argument("--test-data", type=Path, default=Path("datasets/test.csv"),
                    help="Optional test.csv to sanity-check that public LB copy block aligns by date_id.")
    ap.add_argument("--era-col", type=str, default=None)
    ap.add_argument("--target-col", type=str, default=None)
    ap.add_argument("--include-groups", type=str, default=None,
                    help="Comma-separated prefixes from {M,E,I,P,V,S,MOM,D}. Example: M,MOM,V")
    ap.add_argument("--limit-rows", type=int, default=None, help="Optional row cap for quick runs.")

    # Missing-value handling
    ap.add_argument("--impute", type=str, choices=["none", "ffill", "bfill", "ffill_bfill"], default="ffill_bfill",
                    help="Impute numeric features after feature engineering. Default ffill then bfill.")

    # Era bucketing
    ap.add_argument("--era-size", type=int, default=21,
                    help="Bucket size in trading days for grouping date_id into eras (e.g., 21 ~ monthly). "
                         "Use 1 to keep 1-day eras. Use 180 for ~half-year buckets.")
    ap.add_argument("--des-off", action="store_true",
                    help="If set, disable DES by collapsing all rows to a single era.")

    # Feature engineering
    ap.add_argument("--fx-groups", type=str, default="M,E,I,P,V,S,MOM,D",
                    help="Groups to engineer features for (comma separated).")
    ap.add_argument("--fx-cap-cols", type=int, default=None,
                    help="Optional safety cap on how many base features to engineer (first N).")
    ap.add_argument("--dir-agree", type=float, default=None,
                    help="If set, keep only features with directional agreement >= this threshold (e.g., 0.7).")

    # Prediction -> signal for Sharpe
    ap.add_argument("--to-signal-mult", type=float, default=400.0,
                    help="signal = clip(1 + mult * pred, 0, 2) for LastNSharpe.")
    ap.add_argument("--lastn", type=int, default=180,
                    help="Number of last global date_id rows to evaluate on (leak-free).")

    # PB hyperparams
    ap.add_argument("--n-trees", type=int, default=800)
    ap.add_argument("--pack-size", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--lambda-l2", type=float, default=1e-6)
    ap.add_argument("--lambda-dro", type=float, default=0.0)
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--max-bins", type=int, default=8)
    ap.add_argument("--k-cuts", type=int, default=7)
    ap.add_argument("--layer-feature-fraction", type=float, default=0.1)
    ap.add_argument("--direction-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prebinned", action="store_true",
                    help="Treat features as already integer-binned when fitting PackBoost.")

    # Evaluation mode (on bucketed eras)
    ap.add_argument("--cv-splits", type=int, default=0, help="If >0, use TimeSeriesSplit CV over bucketed eras.")
    ap.add_argument("--embargo-eras", type=int, default=0, help="Gap eras removed from end of each train fold (bucketed).")
    ap.add_argument("--holdout-frac", type=float, default=0.05, help="If cv-splits=0 and holdout-eras not set.")
    ap.add_argument("--holdout-eras", type=int, default=1,
                    help="If set (>0), hold out the last N bucketed eras (default 1 bucket).")

    # EFB (PackBoost-only)
    ap.add_argument("--efb-disable", action="store_true",
                    help="Disable Exclusive Feature Bundling for PackBoost path.")
    ap.add_argument("--efb-prefixes", type=str, default="M,E,I,P,V,S,MOM,D",
                    help="Comma-separated prefixes to bundle (default: D). Example: D,M")
    ap.add_argument("--efb-max-conflicts", type=int, default=0,
                    help="Max allowed nonzero overlaps between features in a bundle (train only).")
    ap.add_argument("--efb-max-bundle-size", type=int, default=32,
                    help="Max features per bundle.")

    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    df = pd.read_csv(args.data)
    df = df.replace([np.inf, -np.inf], np.nan)

    include_groups = [g.strip() for g in args.include_groups.split(",")] if args.include_groups else None
    feats, target_raw, era_col_raw = infer_columns(df, args.era_col, args.target_col, include_groups)

    if args.limit_rows is not None and args.limit_rows < len(df):
        df = df.iloc[: args.limit_rows].copy()

    # ---------- Feature engineering ----------
    fx_groups = [g.strip() for g in args.fx_groups.split(",")] if args.fx_groups else []
    df, feats_all = engineer_features(
        df, feats, date_col=era_col_raw, groups_for_fx=fx_groups,
        window_z=21, window_mom=63, cap_cols=args.fx_cap_cols
    )

    # ---------- Optional imputation (after engineering) ----------
    if args.impute != "none":
        num_cols = [c for c, dt in df.dtypes.items() if np.issubdtype(dt, np.number)]
        if args.impute in ("ffill", "ffill_bfill"):
            df[num_cols] = df[num_cols].ffill()
        if args.impute in ("bfill", "ffill_bfill"):
            df[num_cols] = df[num_cols].bfill()

    # ---------- Target enforcement: use excess returns ----------
    target_col = target_raw
    if target_col == "forward_returns":
        if "risk_free_rate" in df.columns:
            df["_excess"] = df["forward_returns"].astype(np.float32) - df["risk_free_rate"].astype(np.float32)
            target_col = "_excess"
            warnings.warn("Converted forward_returns to excess using risk_free_rate into column _excess.")
        else:
            warnings.warn("risk_free_rate missing: using forward_returns (not excess). LastN comparisons may misalign.")

    # We'll also need forward_returns & risk_free_rate for Hull metric
    have_hull_cols = ("forward_returns" in df.columns) and ("risk_free_rate" in df.columns)

    # ---------- Era vectors ----------
    if args.des_off:
        df["_era_bucket"] = 0  # collapse eras => DES off
    else:
        df["_era_bucket"] = bucketize_era(df[era_col_raw], size=int(args.era_size))

    # Features/target arrays
    X_all = df[feats_all].astype(np.float32).values
    y_all = df[target_col].astype(np.float32).values
    era_bucket_all = df["_era_bucket"].astype(np.int32).values

    print(f"Data: rows={len(df):,}, base features={len(feats)}, engineered+base={len(feats_all)}, "
          f"unique raw eras={df[era_col_raw].nunique()}, unique bucketed eras={df['_era_bucket'].nunique()}")
    print(f"Era mode: {'DES-OFF (single era)' if args.des_off else f'bucketed (size={args.era_size})'} | Target: {target_col}")
    if include_groups:
        print(f"Included groups (base): {include_groups}")
    print(f"Engineered on groups: {fx_groups}")
    print(f"Impute: {args.impute}")

    # -------- Temporal holdout (bucketed eras OR chronological if DES-OFF) --------
    eras_unique = df["_era_bucket"].nunique()
    if args.des_off or eras_unique == 1:
        n = len(df)
        n_te = max(1, int(round(args.holdout_frac * n)))
        n_tr = max(1, n - n_te)
        tr_mask = np.zeros(n, dtype=bool)
        tr_mask[:n_tr] = True  # df is already date-sorted in engineer_features
        te_mask = ~tr_mask
        te_eras = np.array([0], dtype=np.int32)
        print(f"DES-OFF row split: train={tr_mask.sum()}, test={te_mask.sum()} (holdout_frac={args.holdout_frac})")
    else:
        tr_mask, te_mask, te_eras = temporal_holdout_masks(
            df["_era_bucket"], holdout_eras=args.holdout_eras, test_frac=args.holdout_frac
        )

    X_tr, X_te = X_all[tr_mask], X_all[te_mask]
    y_tr, y_te = y_all[tr_mask], y_all[te_mask]
    era_tr, era_te = era_bucket_all[tr_mask], era_bucket_all[te_mask]

    # Optional directional-agreement filter (train-only)
    feats_kept = feats_all
    if args.dir_agree is not None:
        keep_idx = dir_agreement_filter(X_tr, y_tr, feats_all, float(args.dir_agree))
        X_tr = X_tr[:, keep_idx]
        X_te = X_te[:, keep_idx]
        X_all = X_all[:, keep_idx]  # keep alignment for global preds
        feats_kept = [feats_all[i] for i in keep_idx]
        print(f"Directional agreement ≥ {args.dir_agree}: kept {len(keep_idx)} features (from {len(feats_all)}).")

    # ---------- PackBoost-only EFB (fit on TRAIN, apply to all PackBoost matrices) ----------
    bundler: Optional[EFBBundler] = None
    if not args.efb_disable:
        efb_prefixes = [p.strip() for p in args.efb_prefixes.split(",") if p.strip()]
        bundler = EFBBundler.fit(
            X_tr, feats_kept, prefixes=efb_prefixes,
            max_conflicts=int(args.efb_max_conflicts),
            max_bundle_size=int(args.efb_max_bundle_size)
        )
        if bundler is not None:
            X_tr_pb = bundler.transform(X_tr)
            X_te_pb = bundler.transform(X_te)
            X_all_pb = bundler.transform(X_all)
            print(f"EFB (PackBoost): bundled {len(feats_kept) - len(bundler.not_sel)} cols into "
                  f"{len(bundler.bundles)} bundles. New PB dims: "
                  f"train={X_tr_pb.shape[1]}, test={X_te_pb.shape[1]}")
        else:
            X_tr_pb, X_te_pb, X_all_pb = X_tr, X_te, X_all
            print("EFB (PackBoost): no eligible columns or no bundling performed.")
    else:
        X_tr_pb, X_te_pb, X_all_pb = X_tr, X_te, X_all
        print("EFB (PackBoost): DISABLED.")

    # Prepare models
    XGBRegressor = _try_import_xgb()
    LGBMRegressor = _try_import_lgbm()
    CatBoostRegressor = _try_import_cat()

    # Helper: predictions -> allocation signal in [0,2]
    def preds_to_signal(preds: np.ndarray) -> np.ndarray:
        return np.clip(1.0 + args.to_signal_mult * preds, 0.0, 2.0)

    # Helper: compute LastNSharpe on last N global date_id rows (exclude training rows -> no leakage)
    def lastn_sharpe_global(preds_all: np.ndarray) -> float:
        mask_last = lastn_mask_by_date(df, era_col_raw, args.lastn)
        mask_eval = mask_last & (~tr_mask)
        if not np.any(mask_eval):
            warnings.warn("No last-N rows left after excluding training rows; returning NaN.")
            return float("nan")
        sig = preds_to_signal(preds_all[mask_eval])

        # Always use true excess if available
        if "forward_returns" in df.columns and "risk_free_rate" in df.columns:
            fex = (df.loc[mask_eval, "forward_returns"].to_numpy(np.float32)
                   - df.loc[mask_eval, "risk_free_rate"].to_numpy(np.float32))
        else:
            fex = y_all[mask_eval]  # fallback (less comparable)
        return annualized_sharpe(sig * fex)

    # Hull metric on last-N global rows (requires forward_returns & rf)
    def hull_penalized_global(preds_all: np.ndarray) -> float | None:
        if not have_hull_cols:
            return None
        mask_last = lastn_mask_by_date(df, era_col_raw, args.lastn)
        mask_eval = mask_last & (~tr_mask)
        if not np.any(mask_eval):
            return None
        sig = preds_to_signal(preds_all[mask_eval])
        fr = df.loc[mask_eval, "forward_returns"].to_numpy(dtype=np.float64)
        rf = df.loc[mask_eval, "risk_free_rate"].to_numpy(dtype=np.float64)
        try:
            return float(hull_penalized_sharpe(sig, fr, rf))
        except Exception as e:
            warnings.warn(f"Failed to compute hull_penalized_sharpe: {e}")
            return None

    results: List[BenchmarkResult] = []

    # -------- PackBoost bench (with EFB features) --------
    pack, rounds = make_packboost(args, args.seed)
    t0 = time.perf_counter()
    era_train = None if args.des_off else era_tr
    pack.fit(X_tr_pb, y_tr, era_train, num_rounds=rounds)
    fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); pb_preds_te = pack.predict(X_te_pb); pred_t = time.perf_counter() - t0
    r2_pb = float(r2_score(y_te, pb_preds_te))
    mc_pb, sh_pb = era_metrics(era_te, y_te, pb_preds_te)
    pb_preds_all = pack.predict(X_all_pb)
    last_pb = lastn_sharpe_global(pb_preds_all)
    hull_pb = hull_penalized_global(pb_preds_all)
    results.append(BenchmarkResult("PackBoost", fit_t, pred_t, r2_pb, mc_pb, sh_pb, last_pb, hull_pb))

    # -------- XGBoost (no EFB; full feature set) --------
    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=args.n_trees, max_depth=args.max_depth, learning_rate=args.learning_rate,
            subsample=1.0, colsample_bytree=1.0, tree_method="hist", max_bin=args.max_bins,
            n_jobs=-1, random_state=args.seed, verbosity=0,
        )
        t0 = time.perf_counter(); xgb.fit(X_tr, y_tr); fit_t = time.perf_counter() - t0
        t0 = time.perf_counter(); xp_te = xgb.predict(X_te); pred_t = time.perf_counter() - t0
        r2_x = float(r2_score(y_te, xp_te)); mc_x, sh_x = era_metrics(era_te, y_te, xp_te)
        xp_all = xgb.predict(X_all)
        last_x = lastn_sharpe_global(xp_all)
        hull_x = hull_penalized_global(xp_all)
        results.append(BenchmarkResult("XGBoost", fit_t, pred_t, r2_x, mc_x, sh_x, last_x, hull_x))
    else:
        print("Skipping XGBoost (not installed).")

    # -------- LightGBM (no EFB; full feature set) --------
    if LGBMRegressor is not None:
        Xtr_df = pd.DataFrame(X_tr); Xte_df = pd.DataFrame(X_te); Xall_df = pd.DataFrame(X_all)
        lgbm = LGBMRegressor(
            n_estimators=args.n_trees, max_depth=args.max_depth, learning_rate=args.learning_rate,
            subsample=1.0, colsample_bytree=1.0, random_state=args.seed, n_jobs=-1, verbose=-1,
            num_leaves=2**args.max_depth, max_bin=args.max_bins,
        )
        t0 = time.perf_counter(); lgbm.fit(Xtr_df, y_tr); fit_t = time.perf_counter() - t0
        t0 = time.perf_counter(); lp_te = lgbm.predict(Xte_df); pred_t = time.perf_counter() - t0
        r2_l = float(r2_score(y_te, lp_te)); mc_l, sh_l = era_metrics(era_te, y_te, lp_te)
        lp_all = lgbm.predict(Xall_df)
        last_l = lastn_sharpe_global(lp_all)
        hull_l = hull_penalized_global(lp_all)
        results.append(BenchmarkResult("LightGBM", fit_t, pred_t, r2_l, mc_l, sh_l, last_l, hull_l))
    else:
        print("Skipping LightGBM (not installed).")

    # -------- CatBoost (no EFB; full feature set) --------
    if CatBoostRegressor is not None:
        cat = CatBoostRegressor(
            iterations=args.n_trees, depth=args.max_depth, learning_rate=args.learning_rate,
            loss_function="RMSE", random_seed=args.seed, verbose=False,
        )
        t0 = time.perf_counter(); cat.fit(X_tr, y_tr); fit_t = time.perf_counter() - t0
        t0 = time.perf_counter(); cp_te = cat.predict(X_te); pred_t = time.perf_counter() - t0
        r2_c = float(r2_score(y_te, cp_te)); mc_c, sh_c = era_metrics(era_te, y_te, cp_te)
        cp_all = cat.predict(X_all)
        last_c = lastn_sharpe_global(cp_all)
        hull_c = hull_penalized_global(cp_all)
        results.append(BenchmarkResult("CatBoost", fit_t, pred_t, r2_c, mc_c, sh_c, last_c, hull_c))
    else:
        print("Skipping CatBoost (not installed).")

    # Optional sanity check if you have test.csv (public LB copy mirrors last 180 date_ids, not buckets)
    if args.test_data.exists():
        try:
            dft = pd.read_csv(args.test_data, usecols=["date_id", "is_scored"])
            print(f"Public test copy sanity check: test rows={len(dft)}, scored rows={int(dft['is_scored'].sum())}")
        except Exception:
            pass

    # -------- Print --------
    print("\nModel         Fit (s)   Predict (s)     R^2   EraMeanCorr   EraSharpe   LastNSharpe   HullPenalized")
    print("-" * 110)
    for r in results:
        hull_str = f"{r.hull_penalized:>13.4f}" if (r.hull_penalized is not None) else f"{'n/a':>13}"
        print(f"{r.name:<12} {r.fit_time:>8.3f} {r.predict_time:>12.3f} {r.r2:>8.4f} "
              f"{r.era_mean_corr:>12.4f} {r.era_sharpe:>11.4f} {r.lastn_sharpe:>13.4f} {hull_str}")


if __name__ == "__main__":
    main()
