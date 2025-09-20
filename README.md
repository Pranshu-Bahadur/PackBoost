# PackBoost

PackBoost is a fast, era-aware gradient boosting library that combines Murky's
ExtraFastBooster design with the Directional Era Splitting (DES) criterion.
It grows *packs* of shallow trees layer-parallel, samples features per depth,
and scores splits using era-bucketed robust gains so that models stay stable
across regime shifts.

## Highlights

- **Pack-parallel boosting** – build `B` trees per round with shared layer-wise
  feature subsets.
- **Directional Era Splitting (DES)** – score candidates with
  `mean − λ·std` across eras plus a directional agreement penalty.
- **Vectorised torch histograms** – milestone 1 ships a pure PyTorch frontier
  that bins per-era statistics in parallel without Python loops.
- **Deterministic by design** – quantile binning, seeded sampling, and pure
  functions keep runs reproducible.
- **Friendly tooling** – scikit-learn compatible wrapper, standalone
  `PackBoostPredictor`, synthetic benchmark, and a Colab-ready Numerai notebook.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
git clone https://github.com/Pranshu-Bahadur/PackBoost.git
cd PackBoost
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

This installs PackBoost in editable mode so the package stays in sync with local
source edits. Append `[numerai]` if you also need the Numerai notebook extras.
CUDA wheels are optional and not required for CPU-only workflows.

### CPU-only quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pytest -q
```

All milestone 1 functionality runs on CPU thanks to the vectorised PyTorch
frontier. No native extensions are required to train or evaluate models.

## Quick Start

```python
from packboost import PackBoost, PackBoostConfig

config = PackBoostConfig(
    pack_size=8,
    max_depth=5,
    learning_rate=0.05,
    lambda_l2=1.0,
    lambda_dro=0.5,
    max_bins=128,
    min_samples_leaf=20,
    random_state=42,
)

booster = PackBoost(config)
booster.fit(X_train, y_train, era_ids_train, num_rounds=20)
preds = booster.predict(X_valid)
```

Monitor correlations during training by providing a validation set and a logging interval:

```python
booster.fit(
    X_train,
    y_train,
    era_ids_train,
    num_rounds=20,
    eval_set=(X_valid, y_valid, era_valid),
    log_evaluation=5,
)
```

### Switching to GPU

```python
config = PackBoostConfig(..., device="cuda")
booster = PackBoost(config)
booster.fit(X_train, y_train, era_ids_train)
```

PackBoost now relies on the native frontier evaluator each depth. It raises a
clear error if `device="cuda"` is requested but the CUDA frontier backend is
missing or a CUDA device is unavailable. You can tune the GPU launch geometry
via `cuda_threads_per_block` and `cuda_rows_per_thread` in `PackBoostConfig` to
better fit your hardware.

## scikit-learn Wrapper

```python
from packboost.wrapper import PackBoostRegressor

model = PackBoostRegressor(pack_size=8, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train, era=era_ids_train)
preds = model.predict(X_valid)
```

## Synthetic Benchmark

The `examples/synthetic_benchmark.py` script compares PackBoost against
XGBoost, LightGBM, and CatBoost with matched hyper-parameters and prints fit /
predict timings plus R² scores.

```bash
python examples/synthetic_benchmark.py
```

Install the optional dependencies (`xgboost`, `lightgbm`, `catboost`) before
running the benchmark.

## Numerai Notebook

A Colab-friendly notebook lives in `notebooks/numerai_gpu_demo.ipynb`. It
illustrates how to:

1. Install dependencies inside Colab (including the native backend)
2. Download the Numerai dataset via `numerapi`
3. Train/evaluate PackBoost on CPU or GPU (Numerai features are already binned)
4. Upload diagnostics back to Numerai

Open the notebook in Google Colab and follow the step-by-step cells. For a local
run, install the optional extras with `pip install -e .[numerai]` to pull in
`numerapi`, `pyarrow`, and related dependencies.

## Testing

```bash
source .venv/bin/activate
pytest -q
```

## Roadmap

- richer GPU kernels (shared-memory histograms, mixed precision)
- configurable era bucketing (quantile & volatility aware)
- Numerai-scale benchmarks with matched split counts

Contributions and benchmarking feedback are welcome.
