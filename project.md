# PackBoost Project Compass

> **Intuition — Why PackBoost? Why combine EFB with DES?**
> ExtraFastBooster (EFB) gives throughput: grow many trees **in lockstep per depth**, repack the frontier to be **feature-major**, and fuse **histogram → scan → scoring → partition**.
> DES (Directional Era Splitting) gives era-robustness: choose splits with **high Newton gain**, **low variance across eras**, and **consistent direction**.
> Together: **fast + era-stable** gradient boosting for Numerai.

---

## 1) Problem Statement & Definitions

We want gradient-boosted trees whose splits are **stable across eras**.

At boosting round $$t$$, we grow a **pack** of $$B$$ trees **synchronously per depth** using the same feature subset for that depth.

**Definition (Pack).**
A *pack* at round $$t$$ is a set $${ f\_{t,1},\dots,f\_{t,B} }$$ of trees grown in parallel that share: (i) the sampled feature subset $$\mathcal{K}\_d$$ at each depth $$d$$, (ii) the frontier scheduling and feature-major repack, and (iii) the DES split selector. Trees diverge as their node partitions evolve differently when splits are applied. Intuitively, a pack is a **mini–random forest embedded into one boosting step** (shared feature subsampling per depth; optional per-tree RNG for rows/thresholds if enabled).

**Pack-average update (decouples step size from $$B$$).**

$$
\hat y^{(t)}(x) \;=\; \hat y^{(t-1)}(x) \;+\; \eta\,\underbrace{\frac{1}{B}\sum_{b=1}^B f_{t,b}(x)}_{\text{pack aggregator }A_t(x)}.
$$

**Loss & gradients.**
Minimize a differentiable loss $$\mathcal{L}(y,\hat y)$$ with per-example
$$g\_i=\frac{\partial \mathcal{L}}{\partial \hat y\_i},\quad h\_i=\frac{\partial^2 \mathcal{L}}{\partial \hat y\_i^2}.$$
**Leaves use Newton updates; DES only affects split choice.**

---

## 2) Mathematical Formulation (DES)

Node sample set $$S$$ is partitioned by era into $${S\_e}$$. Consider a candidate split $$(k,\tau)$$.

### 2.1 Per-era Newton split gain

With L2 $$\lambda\ge0$$ and (optional) complexity $$\gamma\ge0$$,

$$
\phi(G,H)=\tfrac12\,\frac{G^2}{H+\lambda},\qquad
\mathrm{gain}_e = \phi(G_{e,L},H_{e,L}) + \phi(G_{e,R},H_{e,R}) - \phi(G_{e,T},H_{e,T}) - \gamma .
$$

### 2.2 Era aggregation (mean & risk)

For present eras $$\mathcal{E}\_S={e:|S\_e|>0}$$, default **equal-era** weights $$w\_e=1/|\mathcal{E}\_S|$$ (optionally $$w\_e\propto \alpha+|S\_e|$$, $$\alpha\ge0$$):

$$
\mu_G=\sum_{e\in\mathcal{E}_S} w_e\,\mathrm{gain}_e,\qquad
\sigma_G=\sqrt{\sum_{e\in\mathcal{E}_S}w_e\,(\mathrm{gain}_e-\mu_G)^2}.
$$

Exclude absent eras (do **not** inject zeros).

### 2.3 Directional coherence

Global parent value from all eras:

$$
v_T=-\frac{G_T}{H_T+\lambda},\qquad d_T=\mathrm{sign}(v_T).
$$

Per era: $$v\_{e,L}=-\frac{G\_{e,L}}{H\_{e,L}+\lambda}$$, $$v\_{e,R}=-\frac{G\_{e,R}}{H\_{e,R}+\lambda}$$.

Use a smooth proxy for agreement:

$$
\mathcal{D}_s=\sum_{e\in\mathcal{E}_S} w_e\,\tanh(\mathrm{gain}_e/s),\quad s\in[10^{-3},10^{-1}].
$$

(Alternatively, the hard average of indicators $$\in\[0,1]$$.)

### 2.4 DES score (split selector)

$$
\boxed{\ \mathrm{DES}(k,\tau)=\mu_G-\lambda_{\mathrm{dro}}\,\sigma_G+\lambda_{\mathrm{dir}}\,\mathcal{D}_s\ }\quad
(\lambda_{\mathrm{dro}}\ge0,\ \lambda_{\mathrm{dir}}\in\mathbb{R}).
$$

After selecting a split, leaf values are Newton:
$v\_{L/R}=-,\frac{G\_{L/R}}{H\_{L/R}+\lambda}$

---

## 3) Execution Model

1. **Depth-synchronous packs.** At depth $$d$$, sample **one** feature subset $$\mathcal{K}\_d$$ for all $$B$$ trees.
2. **Frontier repack.** Group frontier rows by (node, era) and repack to **feature-major** contiguous spans.
3. **Fused per (node,feature).** Histogram → prefix-scan (all thresholds) → per-era gains → online Welford for $$(\mu\_G,\sigma\_G)$$ + $$\mathcal{D}\_s$$ → pick best $$(k,\tau)$$.
4. **Partition.** Stable segmented scatters produce child shards per (node, era). No host sync on GPU within a depth.

---

## 4) Memory Layout & Workspaces

* **Bins** $$X\_b$$: `uint8` if `max_bins ≤ 255`, else `uint16`, row-major.
* **Row pools:** `int32` contiguous pools per (node, era); SoA offsets per shard.
* **Histogram scratch:** per-(node,feature) `[bins,2]` for $$(\sum g,\sum h)$$ in shared memory (GPU) or thread-local arrays (CPU).
* **Accumulators:** weighted Welford `(count, mean, M2)` for $$(\mu\_G,\sigma\_G)$$; register accumulator for $$\mathcal{D}\_s$$.
* **Tree SoA:** arrays `feature`, `threshold`, `left`, `right`, `leaf_value` per tree.

---

## 5) Backends

**CPU.** OpenMP over (node\_tile, feature\_tile); AVX2/AVX-512 gathers for histogram fills; thread-local histograms → reduce; stable per-era partitions via prefix offsets.

**CUDA.** Grid `(nodes_tile, features_tile)`, block `WARPS_PER_BLOCK × 32`, map **one warp → one (node,feature)**. Each warp:

* accumulates `[bins,2]` into shared memory via warp reductions (avoid atomics),
* runs warp-exclusive scans (shuffles) to produce left aggregates; right by subtraction,
* computes per-era gains and updates online $$(\mu\_G,\sigma\_G)$$ and $$\mathcal{D}\_s$$ in registers,
* performs segmented scatters (warp scans) into persistent workspaces for child partitions.
  Pack dimension $$B$$ is explicit (tile more (node,feature) or include $$B$$ in the tiling domain).

---

## 6) Detailed Algorithm (one boosting round)

**Inputs.** $$X\_b!\in!{0,\dots,\mathrm{bins}-1}^{N\times F}$$, targets $$y$$, predictions $$\hat y^{(t-1)}$$, eras $$e\in{1,\dots,E}$$, config $$(B,D,\lambda,\lambda\_{\mathrm{dro}},\lambda\_{\mathrm{dir}},\alpha,\eta)$$.

**A. Initialize pack and residuals**

1. Compute $$g\_i,h\_i$$ from $$\hat y^{(t-1)}$$.
2. Create $$B$$ empty trees; build `EraShardIndex` (row lists $$S\_e$$) at the root.

**B. Depth loop for $$d=0..D-1$$ (shared across the pack)**

1. Sample one layer feature subset $$\mathcal{K}\_d\subseteq{1,\dots,F}$$.

2. Tile active nodes × $$\mathcal{K}\_d$$ to `(nodes_tile, features_tile)`; stream eras in `era_tile`.

3. Fused scoring per tile:

   * **Histograms:** per-(node,feature,bin,era) accumulate $$(\sum g,\sum h)$$.
   * **Prefix-scan:** exclusive scan over bins → cumulative left $$(G\_{e,L}\[b],H\_{e,L}\[b])$$ for thresholds $$\tau=b$$; right via parent minus left.
   * **Per-era best:** for each era pick $${b\_e^\*}$$ maximizing $$\mathrm{gain}\_e(b)$$; record $${\mathrm{gain}\_e^*}$$.
   * **DES aggregation:** update Welford stats for $$\mu\_G,\sigma\_G$$ over $$gain_e^\*$$; update $$\mathcal{D}_s \mathrel{+}= w_e tanh({\mathrm{gain}_e^*}/s)$$.
   * **Pick split:** select $$(k^\*,\tau^\*)$$ maximizing $$\mathrm{DES}$$ subject to `min_samples_leaf`, `min_child_weight`, `min_split_gain`.

4. **Partition:** segmented stable scatter rows into $$S\_{e,L},S\_{e,R}$$; nodes not splitting become leaves.

5. **Leaf values:** for terminal nodes (or depth cap), set $$v=-G/(H+\lambda)$$.

**C. Pack aggregation and update**

1. $$A\_t(x) = \frac{1}{B}\sum\_{b=1}^B f\_{t,b}(x)$$.
2. $$\hat y^{(t)} \leftarrow \hat y^{(t-1)} + \eta,A\_t(x)$$.
3. Optionally recompute $$(g,h)$$ for the next round.

**D. Output** Serialized $${f\_{t,b}}$$, per-depth $$\mathcal{K}\_d$$, and pack metadata.

---

## 7) Minimal Configuration

```python
@dataclass(frozen=True)
class PackBoostConfig:
    n_estimators: int = 400
    pack_size: int = 8
    max_depth: int = 6
    learning_rate: float = 0.05
    reg_lambda: float = 1.0
    min_samples_leaf: int = 20
    max_bins: int = 64
    layer_feature_fraction: float = 1.0
    # DES
    lambda_dro: float = 0.25
    lambda_dir: float = 0.10
    era_alpha: float = 0.0
    dir_tanh_scale: float = 1.0
    # Execution
    device: str = "cpu"      # "cpu" | "cuda"
    nodes_tile: int = 64
    features_tile: int = 64
    era_tile: int = 64
    seed: int = 42
```

---

## 8) Target Repository Layout

```
packboost/
  __init__.py
  config.py
  data.py
  core.py              # Estimator: fit/predict/save/load; callbacks; logging
  frontier.py          # Dispatch CPU/CUDA; marshal buffers; return partitions
  workspace.py         # Reusable workspaces (CPU/CUDA)
  predict.py           # Packed SoA traversal; optional CUDA traversal
  backends/
    cpu/frontier_cpu.cpp
    cuda/frontier_cuda.cu
    bindings.cpp       # pybind11 surface -> packboost._backend
examples/
benchmarks/
tests/
setup_native.py
```

---

## 9) Milestones (acceptance criteria)

* **M0. Baseline Python frontier (DONE)** — vectorised per-era histograms; Newton split gain; DES aggregation. *Tests:* small synthetic; improves MSE over mean baseline.
* **M1. Correctness hardening (CURRENT)** — use **global parent direction**; add **weighted Welford** (supports `era_alpha>0`); parity vs NumPy reference. *Deliverables:* `tests/test_des.py`, `tests/test_hist.py`. (+Histogram subtraction & sibling reuse — reuse parent hist for children; ablation shows speedup.)
* **M2. CPU backend (OpenMP+SIMD)** — `frontier_cpu.cpp` with per-tile hist + Welford; stable partitions. *Perf:* ≥2× Python frontier on 1e6×256×32.
* **M3. pybind11 wiring** — expose CPU path; CI wheels; Python fallback if backend missing.
* **M4. CUDA v1 (naive)** — warp per (node,feature); global-mem hist; atomics allowed. *Gate:* correctness parity with CPU for ≤64 bins.
* **M5. CUDA v2 (shared-mem hist)** — shared-mem hist + warp reductions (no atomics). *Perf:* ≥4× v1 on A100 (2.7M×2,376×5).
* **M6. CUDA v3 (fused scan+score)** — warp shuffles for scans; inline Welford; segmented scatters. *Perf:* \~200 trees/s at Numerai scale.
* **M7. Precision & memory** — `uint8` bins; `float16` storage with `float32` accumulation; missing-bin column.
* **M8. Auto-tuner** — search `(nodes_tile,features_tile,era_tile,threads/block,rows/thread)`; cache best by dataset signature.
* **M9. Predictor kernels & serialization** — packed SoA traversal (CPU & CUDA); save/load inference bundles.
* **M10. Numerai integration & benchmarks** — end-to-end scripts; throughput and validation metrics logged.

---

## 10) Changelog

- 2025-09-21 — Implemented batched histogram builder and vectorized split scoring for faster Milestone 1 training.
- 2025-09-20 — Applied Milestone 1 audit fixes: global parent direction in DES, weighted Welford with `era_alpha`, histogram subtraction guard, DES regression tests, native backend stubs, and CPU-only documentation updates.
