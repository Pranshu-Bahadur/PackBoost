![PackBoost logo](docs/assets/packboost.png)

**PackBoost** is a high-performance, GPU-accelerated gradient boosting library built with **PyTorch + CUDA** and a **sklearn-like API**. It implements a clean-room version of Murky’s **ExtraFastBooster (EFB)** dataflow (bit-packing, feature sampling, warp-level ops) and integrates **Directional Era Splitting (DES)** (by Timothy DeLise) for era-aware modeling.

* **Extreme speed.** 300–380+ trees/sec at `max_depth=7`, `nfolds=8` on Numerai v5.0 (≈2.7M rows) on an A100.
* **Massive parallelism.** Train up to **32 trees in parallel** via `nfolds`.
* **Era-aware training.** **DES** integrates naturally for time-series / era-stratified data.
* **Strict parity.** CUDA kernels are tested against a vectorized PyTorch CPU reference.
* **Real results.** ~**2.0 Sharpe** on Numerai v5.0 validation with a walk-forward procedure.

> 📒 Results & profiling notebooks
> • Numerai Walkforward Validation results: [https://colab.research.google.com/drive/18X3Psh5ewhSej28QMMSllT6yqmENrO?usp=sharing](https://colab.research.google.com/drive/18X3Psh5ewhSej28QMMSllT6yqmENrO?usp=sharing)
> • Kernel profiling: [https://colab.research.google.com/drive/1MuRymjblt3gDnwydIwfNne34CvF9sgEL?usp=sharing](https://colab.research.google.com/drive/1MuRymjblt3gDnwydIwfNne34CvF9sgEL?usp=sharing)

---

## Installation

**Requirements**

* Python ≥ 3.11
* **PyTorch** with CUDA (tested on 2.x)
* **CUDA toolkit** matching your driver (tested on 12.x)
* **ninja** (recommended for faster builds)
* NVIDIA GPU (A100 tuned; others work but perf varies)
* Windows specific requirements
  * Visual Studio C++ Build Tools (configured by default with default c++ 2022 build tools path)
  * Other versions need to initialize VCVARS64_BAT_FILEPATH with the filepath to vcvars64.bat
    * e.g. `VCVARS64_BAT_FILEPATH="C:\Program Files\Microsoft Visual Studio\{20XX}\BuildTools\VC\Auxiliary\Build\vcvars64.bat"` or `"C:\Program Files (x86)\Microsoft Visual Studio\{20XX}\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`

```bash
# 1) Create a fresh env
python -m venv .venv
source .venv/bin/activate

# 2) Install deps (pin your torch/cuda as needed)
pip install --upgrade pip
pip install git+https://github.com/Pranshu-Bahadur/PackBoost.git
pip install torch --index-url https://download.pytorch.org/whl/cu126 

```

**Notes**

* If you need to set CUDA archs:
  `export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"` (adjust for your GPU).
* First import will trigger a JIT compile of the C++/CUDA extension.

---

## Quickstart (tiny, runs in seconds)

PackBoost expects **int8 binned features** in the range **0–4** (5 bins). Use your own quantizer or a helper.

```python
import numpy as np
from packboost.core import PackBoost   # ensure this matches your public API

# Small toy data for a fast first run
N_train, N_test, F = 10_000, 1_000, 100
X_train = np.random.randint(0, 5, (N_train, F), dtype=np.int8)   # 5 bins (0–4)
y_train = np.random.randn(N_train).astype(np.float32)
X_test  = np.random.randint(0, 5, (N_test,  F), dtype=np.int8)

# Optional: era IDs for DES (e.g., integers by date bucket)
era_ids = np.random.randint(0, 50, N_train).astype(np.int32)

model = PackBoost(
    device='cuda',     # falls back to CPU if CUDA not available
    max_depth=7,
    nfolds=32,         # <-- number of parallel trees
    lr=0.07,
    L2=1e5            
)

model.fit(X_train, y_train, rounds=200, era_ids=None)  # era_ids=None disables DES

pred_q30 = model.predict(X_test)                # int32 in Q30 fixed-point
pred = pred_q30.astype(np.float64) / (1 << 30)  # convert to float
print(pred[:5])
```
---

## API

### `PackBoost.fit(...)`

```python
def fit(self,
        X: np.ndarray, y: np.ndarray,
        Xv: np.ndarray = None, Yv: np.ndarray = None,
        nfolds: int = 8,
        rounds: int = 10_000,
        max_depth: int = 7,
        callbacks: list = None,
        *,
        lr: float = 0.07,
        L2: float = 100_000.0,
        nfeatsets: int = 32,
        qgrad_bits: int = 12,
        seed: int = 42,
        era_ids: np.ndarray | None = None) -> "PackBoost"
```

**Required**

* `X`: `np.int8` array of shape `[N, F]`, **binned features in [0, 4]**.
* `y`: `np.float32` array of shape `[N]`, targets (converted internally to Q30 `int32`).

**Optional / defaults**

* `Xv`, `Yv`: validation features/targets (`int8`/`float32`). Enables online validation updates.
* `nfolds`: **number of parallel trees built each round** (e.g., 8, 16, 32).
* `rounds`: boosting rounds (iterations).
* `max_depth`: tree depth `D` (supports up to 8 currently).
* `callbacks`: list of callables `cb(booster)` run at the end of each round.
* `lr`: learning rate (distributed as `lr / nfolds` internally).
* `L2`: L2 regularization applied in cut selection.
* `nfeatsets`: **feature-set multiplicity per round**; controls how many independent 32-lane bit-schedules you sample (`et_sample_1b`). Shapes: XS is `[nfeatsets, 32*M]`.
* `qgrad_bits`: gradient quantization bits (currently fixed at 12).
* `seed`: RNG for schedules.
* `era_ids`: `np.ndarray[int]` of length `N` (sorted by time). When provided, **DES** path is used (era boundaries are inferred; single-era if omitted).

**Returns**: the fitted `PackBoost` instance.

**Constraints & behavior**

* **Dtypes are enforced**: `assert X.dtype == np.int8 and y.dtype == np.float32` (same for `Xv`, `Yv`).
* Bit-packing requires `4*F < 2**16` (schedule dtype is `uint16`).
* Device selection: if `device='cuda'` but CUDA is unavailable, training falls back to CPU.
* **DES** uses computed `era_ends`; non-DES path treats the dataset as a single era.
* Predictions are maintained in **Q30** fixed-point (`int32`). Convert via `pred / (1<<30)`.

**Example with validation + callback**

```python
class LoggingCallback:
    def __init__(self, every=500): self.every = every
    def __call__(self, booster):
        t = booster.tree_set
        if t % self.every == 0:
            # Example: compute a simple running metric on validation if present
            if hasattr(booster, "Pv_") and hasattr(booster, "Yv"):
                pv = booster.Pv_.detach().cpu().numpy().astype(np.float64) / (1 << 30)
                yv = booster.Yv.detach().cpu().numpy()
                print(f"[round {t}] val_mean={pv.mean():.6f}")

model.fit(
    X_train, y_train,
    Xv=X_val, Yv=y_val,
    nfolds=32, rounds=2000, max_depth=7,
    lr=0.07, L2=1e5, nfeatsets=32,
    era_ids=era_ids,
    callbacks=[LoggingCallback(every=250)]
)
```

### `PackBoost.predict(X) -> np.ndarray[int32]`

* Input `X`: `np.int8` in `[0,4]`, shape `[N, F]`.
* Returns: `int32` Q30 predictions; convert with `/ (1<<30)`.

---

## Benchmarks

**Hardware/Software:** NVIDIA A100 (80GB), CUDA 12.x, PyTorch 2.x, Ubuntu 22.04
**Dataset:** Numerai v5.0 training (≈2.7M rows), `K=8`, `max_depth=7`
**Method:** Per-kernel averages across rounds; see profiling notebook for exact repro.

| Kernel / Stage        | PackBoost (C++/PyTorch) |   EFB (Numba) |  Speedup (EFB ÷ PB) |
| --------------------- | ----------------------: | ------------: | ------------------: |
| `prep_vars`           |                2.214 ms |      1.500 ms | 0.68× *(PB slower)* |
| `repack`              |                0.736 ms |      3.212 ms |           **4.36×** |
| `et_sample_1b`        |                0.619 ms |      1.139 ms |           **1.84×** |
| `H`                   |               15.623 ms |     18.370 ms |           **1.18×** |
| `H0`                  |                1.360 ms |      2.688 ms |           **1.98×** |
| `cut`                 |                0.009 ms |      0.089 ms |           **9.89×** |
| `advance_and_predict` |                0.405 ms |      0.619 ms |           **1.53×** |
| **Total / round**     |           **20.965 ms** | **27.617 ms** |           **1.32×** |

**Estimated Trees per Second (TPS)**

* **PackBoost:** **381.6** trees/s *(K=8, 20.97 ms/round)*
* **ExtraFastBooster:** **289.7** trees/s *(K=8, 27.62 ms/round)*

> These numbers come from the attached profiling run and the linked notebook; expect variation across hardware/toolchain.

---

## Current Limitations & Roadmap

* **Feature bins:** encoding path assumes **5 bins (0–4)**. Roadmap: up to **127 bins**.
* **GPU tuning:** kernels tuned on **A100**; performance on other GPUs may vary.
* **Gradient quantization:** `qgrad_bits=12` fixed for now; parameterization planned.
* **Max depth:** supported up to **8**; **9** in development.

---

## Testing

```bash
pip install pytest
pytest -q tests/test_cuda_kernels.py
# Parity tests compare CUDA kernels vs the vectorized CPU reference.
```

---

## Troubleshooting

* **Build fails on first import** → ensure CUDA toolkit matches driver; install `ninja`.
* **Wrong CUDA arch** → set `TORCH_CUDA_ARCH_LIST` (e.g., `8.0;8.6;8.9`).
* **Shape/index errors** → feature count and bin count must match the model’s training config.

---

## References & Acknowledgements

* Murky, **“200 trees per second with CUDA (ExtraFastBooster)”**
  [https://forum.numer.ai/t/200-trees-per-second-with-cuda/7823](https://forum.numer.ai/t/200-trees-per-second-with-cuda/7823)
* Timothy DeLise, **“Directional Era Splitting (DES)”**
  [https://www.sciencedirect.com/science/article/abs/pii/S0957417425033020](https://www.sciencedirect.com/science/article/abs/pii/S0957417425033020)

Thanks to Murky for the core EFB concepts and to Timothy DeLise for DES. PackBoost is a clean-room implementation inspired by those ideas.

---

## License

**GPLv3** — see `LICENSE`.