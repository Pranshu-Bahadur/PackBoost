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
- **CPU & CUDA frontier backends** – native C++/CUDA extensions batch whole
  depth frontiers, score DES splits, and return child partitions without Python
  loops.
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
pip install -e .[cuda]  # add [cuda] if you plan to build on GPU
# Optionally include [numerai] to install Numerai notebook dependencies
```

### Native backends

PackBoost ships optional C++/CUDA extensions that provide high-performance
histogram builders. Build them with `pybind11` and a modern compiler:

```bash
pip install pybind11 numpy
python setup_native.py build_ext --inplace
```

CUDA users need `nvcc` in `PATH`. The build script detects it automatically and
compiles the optimized kernels (including the frontier evaluator) when available.
Set `PACKBOOST_DISABLE_CUDA=1` before the build to force a CPU-only wheel even on
GPU machines. Without CUDA the script still builds the fast multi-threaded CPU
backend; if neither backend is built PackBoost falls back to the pure NumPy
implementation.

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

PackBoost now relies on the native frontier evaluator each depth. It raises a
clear error if `device="cuda"` is requested but the CUDA frontier backend is
missing or a CUDA device is unavailable.

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
