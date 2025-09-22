# examples/hull_benchmark.py
from __future__ import annotations
import sys, time, argparse
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
    raise RuntimeError("PackBoost not found on PYTHONPATH. Install/`pip -e .` your package.") from e


@dataclass
class BenchmarkResult:
    name: str
    fit_time: float
    predict_time: float
    r2: float
    era_mean_corr: float
    era_sharpe: float
    lastn_sharpe: float


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
    cors = df.groupby("era").apply(lambda g: _safe_corr(g["y"].values, g["p"].values)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(cors) == 0:
        return float("nan"), float("nan")
    mean = float(cors.mean())
    std = float(cors.std(ddof=0)) if len(cors) > 1 else np.nan
    sharpe = float(mean / std) if (std is not None and std > 0) else np.nan
    return mean, sharpe

def last_n_sharpe(y: np.ndarray, p: np.ndarray, n: int = 180, mode: str = "sign", annualize: int = 0) -> float:
    """
    Sharpe over the last N samples of the provided arrays.
    mode:
      - 'sign' (default): weights = sign(pred), scale-free long/short
      - 'rank'         : weights in [-1,1] from percentile rank
      - 'raw'          : weights = z-scored pred (scale-normalized)
    annualize: if >0, multiply by sqrt(annualize); else leave as pure sample Sharpe.
    """
    y = np.asarray(y); p = np.asarray(p)
    n = max(1, min(int(n), len(y)))
    y = y[-n:]; p = p[-n:]
    if mode == "sign":
        w = np.sign(p)
    elif mode == "rank":
        r = pd.Series(p).rank(pct=True).to_numpy()
        w = (r - 0.5) * 2.0
    else:  # 'raw'
        z = (p - p.mean()) / (p.std(ddof=0) + 1e-12)
        w = z
    ret = y * w
    mu = ret.mean()
    sd = ret.std(ddof=0)
    sh = mu / (sd + 1e-12)
    if annualize and np.isfinite(sh):
        sh *= np.sqrt(annualize)
    return float(sh)


# ---------- Column inference & feature groups ----------
GROUP_PREFIXES = ("M", "E", "I", "P", "V", "S", "MOM", "D")

def infer_columns(df: pd.DataFrame, era_col_arg: Optional[str], target_col_arg: Optional[str],
                  include_groups: Optional[List[str]]) -> Tuple[List[str], str, str]:
    # Era
    era_col = era_col_arg or ("date_id" if "date_id" in df.columns else None)
    if era_col is None:
        raise ValueError("Couldn't find an era column. Pass --era-col or include 'date_id' in CSV.")

    # Target (train only)
    if target_col_arg and target_col_arg in df.columns:
        target_col = target_col_arg
    else:
        candidates = [c for c in ["market_forward_excess_returns", "forward_returns", "target", "y"] if c in df.columns]
        if not candidates:
            raise ValueError("Couldn't find a target. Pass --target-col or include one of "
                             "['market_forward_excess_returns','forward_returns','target','y'].")
        target_col = candidates[0]

    # Features: numeric columns excluding meta; optionally filter by group prefixes
    meta = {era_col, target_col, "risk_free_rate"}
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
    """
    Group consecutive date_id values into fixed-size buckets.
    Example: size=21 ~ 'monthly-ish' trading days.
    """
    if size <= 1:
        return era_series.astype(np.int32)
    base = int(era_series.min())
    return (((era_series.astype(np.int64) - base) // int(size))).astype(np.int32)


# ---------- Holdout by daily days (public-LB mimic) ----------
def temporal_holdout_masks_by_days(date_id_series: pd.Series, holdout_days: int = 180):
    uniq_days = np.array(sorted(date_id_series.unique()))
    k = min(int(holdout_days), max(1, len(uniq_days) - 1))
    test_days = set(uniq_days[-k:])
    tr = ~date_id_series.isin(test_days)
    te =  date_id_series.isin(test_days)
    return tr.values, te.values, uniq_days[-k:]


# ---------- Fast CV over daily days ----------
def cv_splits_by_days(date_id_series: pd.Series, n_splits=5, embargo_days=0):
    uniq_days = np.array(sorted(date_id_series.unique()))
    day_codes = pd.Categorical(date_id_series, categories=uniq_days, ordered=True).codes
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr_day_idx, te_day_idx in tss.split(uniq_days):
        if embargo_days > 0 and len(tr_day_idx) > embargo_days:
            tr_day_idx = tr_day_idx[:-embargo_days]
        tr_mask = np.in1d(day_codes, tr_day_idx, assume_unique=True)
        te_mask = np.in1d(day_codes, te_day_idx, assume_unique=True)
        yield tr_mask, te_mask


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
    )
    if (args.n_trees % cfg.pack_size) != 0:
        raise ValueError("n_trees must be divisible by pack_size.")
    rounds = args.n_trees // cfg.pack_size
    booster = PackBoost(cfg)
    return booster, rounds

def bench_one(
    name: str,
    fit_fn: Callable[[], None],
    pred_fn: Callable[[], np.ndarray],
    y_true: np.ndarray,
    era_test: np.ndarray,
    lastn: int,
    sharpe_mode: str,
    annualize: int,
) -> BenchmarkResult:
    t0 = time.perf_counter(); fit_fn(); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); preds = pred_fn(); pred_t = time.perf_counter() - t0
    r2 = float(r2_score(y_true, preds))
    mc, sh = era_metrics(era_test, y_true, preds)
    ln_sh = last_n_sharpe(y_true, preds, n=lastn, mode=sharpe_mode, annualize=annualize)
    return BenchmarkResult(name, fit_t, pred_t, r2, mc, sh, ln_sh)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Hull benchmark: PackBoost vs baselines with era-aware eval.")
    ap.add_argument("--data", type=Path, default=Path("datasets/train.csv"))
    ap.add_argument("--era-col", type=str, default=None)
    ap.add_argument("--target-col", type=str, default=None)
    ap.add_argument("--include-groups", type=str, default=None,
                    help="Comma-separated prefixes from {M,E,I,P,V,S,MOM,D}. Example: M,MOM,V")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with NaNs/Infs (default: keep).")
    ap.add_argument("--limit-rows", type=int, default=None, help="Optional row cap for quick runs.")

    # Era bucketing (for DES/metrics only)
    ap.add_argument("--era-size", type=int, default=180,
                    help="Bucket size (trading days) for grouping date_id into eras. 21≈monthly, 180≈half-year, 1=no bucketing.")

    # PB hyperparams
    ap.add_argument("--n-trees", type=int, default=2000)
    ap.add_argument("--pack-size", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--lambda-l2", type=float, default=1e-6)
    ap.add_argument("--lambda-dro", type=float, default=0.0)
    ap.add_argument("--min-samples-leaf", type=int, default=20)
    ap.add_argument("--max-bins", type=int, default=16)
    ap.add_argument("--k-cuts", type=int, default=4)
    ap.add_argument("--layer-feature-fraction", type=float, default=0.5)
    ap.add_argument("--direction-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # Evaluation controls
    ap.add_argument("--cv-splits", type=int, default=0, help="If >0, use TimeSeriesSplit CV over *daily* date_ids.")
    ap.add_argument("--embargo-days", type=int, default=0, help="Gap (trading days) removed from end of each train fold.")
    ap.add_argument("--holdout-days", type=int, default=180, help="Hold out the last N unique date_ids (default 180).")

    # Last-N Sharpe config
    ap.add_argument("--lastn-sharpe", type=int, default=180, help="N for LastNSharpe column (default 180).")
    ap.add_argument("--sharpe-mode", type=str, default="sign", choices=["sign", "rank", "raw"],
                    help="Sharpe weighting: sign (scale-free, default), rank (bounded), raw (z-scored preds).")
    ap.add_argument("--annualize", type=int, default=0, help="If >0, annualize Sharpe by sqrt(value), e.g. 252.")

    # Which baselines to run (comma list). Default empty -> PackBoost only (fast).
    ap.add_argument("--baselines", type=str, default="",
                    help="Comma separated subset of {xgb,lgbm,cat}. Example: --baselines xgb,lgbm")

    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    df = pd.read_csv(args.data)
    df = df.replace([np.inf, -np.inf], np.nan)
    if args.dropna:
        df = df.dropna()

    include_groups = [g.strip() for g in args.include_groups.split(",")] if args.include_groups else None
    feats, target, era_col_raw = infer_columns(df, args.era_col, args.target_col, include_groups)

    if args.limit_rows is not None and args.limit_rows < len(df):
        df = df.iloc[: args.limit_rows].copy()

    # Era bucketing for DES/metrics
    era_source_col = "date_id" if "date_id" in df.columns else era_col_raw
    df["_era_bucket"] = bucketize_era(df[era_source_col], size=int(args.era_size))

    # Arrays
    X_all = df[feats].astype(np.float32).values
    y_all = df[target].astype(np.float32).values
    era_bucket_all = df["_era_bucket"].astype(np.int32).values
    date_id_series = df["date_id"] if "date_id" in df.columns else df[era_col_raw]

    print(f"Data: rows={len(df):,}, features={len(feats)}, "
          f"unique daily days={date_id_series.nunique()}, bucketed eras={df['_era_bucket'].nunique()}")
    print(f"Eras: bucket size={args.era_size} days  | Target: {target}")
    if include_groups:
        print(f"Included groups: {include_groups}")

    # Baselines to run
    want = {x.strip().lower() for x in args.baselines.split(",")} if args.baselines else set()
    XGBRegressor = _try_import_xgb() if ("xgb" in want) else None
    LGBMRegressor = _try_import_lgbm() if ("lgbm" in want) else None
    CatBoostRegressor = _try_import_cat() if ("cat" in want) else None

    results: List[BenchmarkResult] = []

    # -------- CV mode (fast, over daily date_ids) --------
    if args.cv_splits and args.cv_splits > 0:
        print(f"\nUsing TimeSeriesSplit CV on daily date_ids: splits={args.cv_splits}, embargo_days={args.embargo_days}\n")
        fold = 0
        def _acc(d: dict[str, list], k: str, v: float): d.setdefault(k, []).append(v)
        agg: dict[str, dict[str, list]] = {}

        for tr_mask, te_mask in cv_splits_by_days(date_id_series, n_splits=args.cv_splits, embargo_days=args.embargo_days):
            fold += 1
            X_tr, X_te = X_all[tr_mask], X_all[te_mask]
            y_tr, y_te = y_all[tr_mask], y_all[te_mask]
            era_tr, era_te = era_bucket_all[tr_mask], era_bucket_all[te_mask]

            # PackBoost
            pack, rounds = make_packboost(args, args.seed + fold)
            def _pb_fit(): pack.fit(X_tr, y_tr, era_tr, num_rounds=rounds)
            def _pb_pred(): return pack.predict(X_te)
            r = bench_one("PackBoost", _pb_fit, _pb_pred, y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize)
            agg.setdefault("PackBoost", {"fit":[], "pred":[], "r2":[], "mc":[], "es":[], "ls":[]})
            _acc(agg["PackBoost"], "fit", r.fit_time); _acc(agg["PackBoost"], "pred", r.predict_time)
            _acc(agg["PackBoost"], "r2", r.r2); _acc(agg["PackBoost"], "mc", r.era_mean_corr)
            _acc(agg["PackBoost"], "es", r.era_sharpe); _acc(agg["PackBoost"], "ls", r.lastn_sharpe)

            # Baselines (optional)
            if XGBRegressor is not None:
                xgb = XGBRegressor(
                    n_estimators=args.n_trees, max_depth=args.max_depth, learning_rate=args.learning_rate,
                    subsample=1.0, colsample_bytree=1.0, tree_method="hist", max_bin=args.max_bins,
                    n_jobs=-1, random_state=args.seed + fold, verbosity=0,
                )
                r = bench_one("XGBoost", lambda: xgb.fit(X_tr, y_tr), lambda: xgb.predict(X_te),
                              y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize)
                agg.setdefault("XGBoost", {"fit":[], "pred":[], "r2":[], "mc":[], "es":[], "ls":[]})
                _acc(agg["XGBoost"], "fit", r.fit_time); _acc(agg["XGBoost"], "pred", r.predict_time)
                _acc(agg["XGBoost"], "r2", r.r2); _acc(agg["XGBoost"], "mc", r.era_mean_corr)
                _acc(agg["XGBoost"], "es", r.era_sharpe); _acc(agg["XGBoost"], "ls", r.lastn_sharpe)

            if LGBMRegressor is not None:
                lgbm = LGBMRegressor(
                    n_estimators=args.n_trees, max_depth=args.max_depth, learning_rate=args.learning_rate,
                    subsample=1.0, colsample_bytree=1.0, random_state=args.seed + fold, n_jobs=-1, verbose=-1,
                    num_leaves=2**args.max_depth, max_bin=args.max_bins,
                )
                r = bench_one("LightGBM", lambda: lgbm.fit(X_tr, y_tr), lambda: lgbm.predict(X_te),
                              y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize)
                agg.setdefault("LightGBM", {"fit":[], "pred":[], "r2":[], "mc":[], "es":[], "ls":[]})
                _acc(agg["LightGBM"], "fit", r.fit_time); _acc(agg["LightGBM"], "pred", r.predict_time)
                _acc(agg["LightGBM"], "r2", r.r2); _acc(agg["LightGBM"], "mc", r.era_mean_corr)
                _acc(agg["LightGBM"], "es", r.era_sharpe); _acc(agg["LightGBM"], "ls", r.lastn_sharpe)

            if CatBoostRegressor is not None:
                cat = CatBoostRegressor(
                    iterations=args.n_trees, depth=args.max_depth, learning_rate=args.learning_rate,
                    loss_function="RMSE", random_seed=args.seed + fold, verbose=False,
                )
                r = bench_one("CatBoost", lambda: cat.fit(X_tr, y_tr), lambda: cat.predict(X_te),
                              y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize)
                agg.setdefault("CatBoost", {"fit":[], "pred":[], "r2":[], "mc":[], "es":[], "ls":[]})
                _acc(agg["CatBoost"], "fit", r.fit_time); _acc(agg["CatBoost"], "pred", r.predict_time)
                _acc(agg["CatBoost"], "r2", r.r2); _acc(agg["CatBoost"], "mc", r.era_mean_corr)
                _acc(agg["CatBoost"], "es", r.era_sharpe); _acc(agg["CatBoost"], "ls", r.lastn_sharpe)

        def _avg(xs): return float(np.mean(xs)) if xs else float("nan")
        for name, m in agg.items():
            results.append(BenchmarkResult(
                name=name,
                fit_time=_avg(m["fit"]), predict_time=_avg(m["pred"]), r2=_avg(m["r2"]),
                era_mean_corr=float(np.nanmean(m["mc"])) if m["mc"] else float("nan"),
                era_sharpe=float(np.nanmean(m["es"])) if m["es"] else float("nan"),
                lastn_sharpe=float(np.nanmean(m["ls"])) if m["ls"] else float("nan"),
            ))

    # -------- Temporal holdout (last N daily samples, default 180) --------
    else:
        tr_mask, te_mask, te_days = temporal_holdout_masks_by_days(date_id_series, holdout_days=args.holdout_days)
        X_tr, X_te = X_all[tr_mask], X_all[te_mask]
        y_tr, y_te = y_all[tr_mask], y_all[te_mask]
        era_tr, era_te = era_bucket_all[tr_mask], era_bucket_all[te_mask]

        print(f"\nUsing temporal holdout by daily date_id. "
              f"Test samples (days): {len(te_days)} (first={te_days[0]}, last={te_days[-1]})\n")

        # PackBoost
        pack, rounds = make_packboost(args, args.seed)
        def _pb_fit(): pack.fit(X_tr, y_tr, era_tr, num_rounds=rounds)
        def _pb_pred(): return pack.predict(X_te)
        results.append(bench_one("PackBoost", _pb_fit, _pb_pred, y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize))

        # Baselines (optional)
        want = {x.strip().lower() for x in args.baselines.split(",")} if args.baselines else set()
        if "xgb" in want and (xb := _try_import_xgb()) is not None:
            xgb = xb(
                n_estimators=args.n_trees, max_depth=args.max_depth, learning_rate=args.learning_rate,
                subsample=1.0, colsample_bytree=1.0, tree_method="hist", max_bin=args.max_bins,
                n_jobs=-1, random_state=args.seed, verbosity=0,
            )
            results.append(bench_one("XGBoost", lambda: xgb.fit(X_tr, y_tr), lambda: xgb.predict(X_te),
                                     y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize))
        if "lgbm" in want and (lb := _try_import_lgbm()) is not None:
            lgbm = lb(
                n_estimators=args.n_trees, max_depth=args.max_depth, learning_rate=args.learning_rate,
                subsample=1.0, colsample_bytree=1.0, random_state=args.seed, n_jobs=-1, verbose=-1,
                num_leaves=2**args.max_depth, max_bin=args.max_bins,
            )
            results.append(bench_one("LightGBM", lambda: lgbm.fit(pd.DataFrame(X_tr, columns=feats), y_tr),
                                     lambda: lgbm.predict(pd.DataFrame(X_te, columns=feats)),
                                     y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize))
        if "cat" in want and (cb := _try_import_cat()) is not None:
            cat = cb(
                iterations=args.n_trees, depth=args.max_depth, learning_rate=args.learning_rate,
                loss_function="RMSE", random_seed=args.seed, verbose=False,
            )
            results.append(bench_one("CatBoost", lambda: cat.fit(X_tr, y_tr), lambda: cat.predict(X_te),
                                     y_te, era_te, args.lastn_sharpe, args.sharpe_mode, args.annualize))

    # -------- Print --------
    print("\nModel         Fit (s)   Predict (s)     R^2   EraMeanCorr   EraSharpe   LastNSharpe")
    print("-" * 86)
    for r in results:
        print(f"{r.name:<12} {r.fit_time:>8.3f} {r.predict_time:>12.3f} {r.r2:>8.4f} "
              f"{r.era_mean_corr:>12.4f} {r.era_sharpe:>11.4f} {r.lastn_sharpe:>13.4f}")


if __name__ == "__main__":
    main()
