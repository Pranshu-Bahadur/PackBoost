# PackBoost

PackBoost is a fast, era-aware gradient boosting library that marries Murky's
ExtraFastBooster design with the Directional Era Splitting (DES) criterion.
It grows *packs* of shallow trees layer-parallel, samples features per depth,
and scores splits using era-bucketed robust gains so that models stay stable
across regime shifts.

## Highlights

- **Pack-parallel boosting** – build `B` trees per round with shared layer-wise
  feature subsets.
- **Directional Era Splitting (DES)** – score candidates with
  `mean − λ·std` across eras plus a directional agreement penalty.
- **CPU & CUDA backends** – fast NumPy implementation out of the box, with an
  optional CuPy-powered GPU path when a CUDA device is available.
- **Deterministic by design** – quantile binning, seeded sampling, and pure
  functions keep runs reproducible.
- **Friendly tooling** – scikit-learn compatible wrapper, standalone
  `PackBoostPredictor`, synthetic benchmark, and a Colab-ready Numerai notebook.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # create this if you use pinned deps
pip install numpy scikit-learn lightgbm xgboost catboost
```

To enable GPU acceleration install CuPy compiled for your CUDA toolkit, e.g.:

```bash
pip install cupy-cuda12x  # replace with the wheel matching your CUDA version
```

If CuPy or a GPU is not present PackBoost will fall back to the CPU backend.

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

### Switching to GPU

```python
config = PackBoostConfig(..., device="cuda")
booster = PackBoost(config)
booster.fit(X_train, y_train, era_ids_train)
```

PackBoost will raise a clear error if `device="cuda"` is requested but CuPy or a
CUDA device is unavailable.

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

1. Install dependencies inside Colab
2. Download the Numerai dataset via `numerapi`
3. Bin features deterministically
4. Train/evaluate PackBoost on CPU or GPU
5. Upload diagnostics back to Numerai

Open the notebook in Google Colab and follow the step-by-step cells.

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
