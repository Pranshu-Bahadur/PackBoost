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

> **DES-off shorthand.** Passing `era_ids=None` to `PackBoost.fit` now collapses the round to a single synthetic era while keeping the frontier batching and instrumentation intact—handy for ablation runs or Hull's DES-off mode.

---

## 4) Memory Layout & Workspaces

* **Bins** $$X\_b$$: `uint8` if `max_bins ≤ 255`, else `uint16`, row-major.
* **Prebinned fast-path:** set `PackBoostConfig.prebinned=True` when the caller supplies integer bins in `[0, max_bins)` to skip quantile preprocessing.
* **Row pools:** `int32` contiguous pools per (node, era); SoA offsets per shard.
* **Histogram scratch:** per-(node,feature) `[bins,2]` for $$(\sum g,\sum h)$$ in shared memory (GPU) or thread-local arrays (CPU).
* **Accumulators:** weighted Welford `(count, mean, M2)` for $$(\mu\_G,\sigma\_G)$$; register accumulator for $$\mathcal{D}\_s$$.
* **Tree SoA:** arrays `feature`, `threshold`, `left`, `right`, `leaf_value` per tree.

---

## 5) Backends

**CPU.** OpenMP over (node\_tile, feature\_tile); AVX2/AVX-512 gathers for histogram fills; thread-local histograms → reduce; stable per-era partitions via prefix offsets.

* **Root histogram reuse.** Identical node shards (e.g., the pack roots at depth 0 or synchronized children) are detected via row signatures so histograms and DES scans are built once and decisions broadcast to duplicates. The native backend mirrors the Python frontier to keep instrumentation aligned.
* **Directional parity with CUDA blueprint.** Per-era leaf values now drive a hard direction signal (`+1` if left > right else `-1`), matching the `directional_split_kernel` contract for future GPU integration.

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
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PackBoostConfig:
    pack_size: int = 8
    max_depth: int = 6
    learning_rate: float = 0.1
    lambda_l2: float = 1e-6
    lambda_dro: float = 0.0
    direction_weight: float = 0.0
    min_samples_leaf: int = 20
    max_bins: int = 63
    k_cuts: int = 0  # 0 => use all (bins-1) thresholds
    cut_selection: Literal["even", "mass"] = "even"
    layer_feature_fraction: float = 1.0
    era_alpha: float = 0.0
    era_tile_size: int = 32
    histogram_mode: Literal["rebuild", "subtract", "auto"] = "subtract"
    feature_block_size: int = 32
    enable_node_batching: bool = True
    random_state: int | None = None
    device: str = "cpu"  # "cpu" | "cuda"
    prebinned: bool = False
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

### M1 Foundations — Final (On-the-fly K-cuts + Subtract-First Frontier) (DONE)

**Numerical policy.** Counts/offsets: **int64**. Node/feature ids & thresholds: **int32**. Bins: `uint8/uint16`. Stats: **float32**.

**Histogram strategy.**

* Modes: `rebuild | subtract | auto`.
  `auto` = **subtract-first + validator**; failing nodes rebuild right via reverse-cumsum.
* **One depth-concat** of `(rows, era_ids, node_ids, grad, hess)` → reused for all feature tiles at that depth.
* Per feature-block histograms over `(node, era, bin)` via shared `bincount`.

**On-the-fly K-cuts (thermometer lanes).**

* Evaluate only **K** candidate thresholds per feature at each depth (instead of all `bins−1`):

  * `cut_selection = "even"`: evenly spaced bin edges.
  * `cut_selection = "mass"`: per-feature mass quantiles from the block histogram.
* Gather left prefixes at the chosen cuts; **right = total − left** (or rebuild in `rebuild` mode).
* Complexity scales with **K**, not `bins−1`. Works for any `max_bins ≤ 256`.

**Adaptive-K policy (depth-synchronous).**

* `k_cuts = 0` ⇒ full sweep (parity mode).
* Otherwise choose **K per depth** from median node size (large/medium/small tiers) with optional depth decay (`k_depth_decay`), bounded by `k_cuts`.
* Preserves pack/EFB behavior: **shared feature subset and shared K** across the pack at a given depth.

**Scans & scoring (DES).**

* Era-wise exclusive prefixes for left; **right by subtraction**.
* Aggregate prefixes across eras via `sum(dim=era)`.
* **Pure DES (Welford):** score = `mean_gain − λ_dro · std_gain` (equal-era by default; optional `era_alpha>0` to bias by era mass).
  Directional term is optional (`direction_weight`, default 0).
* Global validity guard on aggregated child counts; per-era guards for gain; **stable tie-break:** `(gain ↓, sample_count ↓, feature_id ↑, threshold ↑)`.

**Pack growth & update.**

* **Depth-synchronous packs** (EFB-style): one sampled feature subset per depth, shared across the B trees.
* After splits, update leaves with Newton values; predictions updated with **pack-average** (`learning_rate / pack_size`).

**Prediction.**

* Trees compile once per device to flat SoA tensors; routing is fully vectorized.
* Optional `predict_packwise(block_size_trees)` batches trees for higher throughput with parity.

**Instrumentation.**

* Per-depth JSON log includes: `hist_ms`, `scan_ms`, `score_ms`, `partition_ms`, `feature_block_size`,
  `nodes_processed`, `nodes_rebuild`, `nodes_subtract_ok`, `nodes_subtract_fallback`, and **`k_cuts_effective`**.

**Acceptance checks.**

* **Parity mode:** with `k_cuts=0` and `histogram_mode="subtract"` (or `"auto"` without validator hits),
  splits and predictions match the baseline within ≤1e−7 on 4 datasets × 5 seeds.
* **Histogram accounting:** at every depth, `nodes_subtract_ok + nodes_rebuild == nodes_processed`.
* **Complexity sanity:** with `max_bins` fixed, observed `score_ms` reduces roughly ∝ `K / (bins−1)` when using K-cuts.
* **Perf floor:** `(hist_ms + scan_ms)` improvement ≥20% vs the pre-M1 Python frontier on the medium synthetic benchmark.

---

### M2. Native CPU backend (OpenMP + SIMD)

#### Objective

Implement a drop-in CPU backend that matches the Torch path’s math (K-cuts + DES with weighted Welford) but runs natively using **OpenMP** for threading and **SIMD (AVX2/AVX-512)** for vectorization. It must integrate with the current Python API and pass parity tests against the Torch implementation.

---

#### Scope (what to build)

1. **Core split finder (CPU)**
   `find_best_splits_batched_cpu(...)` that mirrors `_find_best_splits_batched`:

   * Inputs: `bins[N,F] (uint8)`, `grad[N] (float32 or int16)`, `era_ids[N] (int16)`, active `nodes` (per-era row indices), `feature_subset`, `max_bins`, `k_cuts`, `cut_selection`, `lambda_l2`, `lambda_dro`, `direction_weight`, `min_samples_leaf`.
   * Output: per active node → `(feature, threshold, score, left_grad, left_hess, right_grad, right_hess, left_count, right_count, left_rows, right_rows)` (same struct as now; rows can be returned as indices or defer partitioning to GPU path—pick one and keep consistent).

2. **Histogram + DES pipeline (CPU)**

   * Build **counts (int32)** and **grad (int32/float)** histograms per *(feature block, node, era)*.
   * **No separate Hessian hist** (for squared loss: `H ≡ count`).
   * Prefix-sum (≤t) across bins.
   * **K-cuts**: even or mass (quantile by counts).
   * **DES**: per-era gains streamed into **weighted Welford** (keep only `(wsum, mean, M2)` in memory).
   * **Right** stats via `total − left` (no second cumsum).

3. **Bindings & switch**

   * `pybind11` wrapper exposed as `packboost_cpu.find_best_splits_batched(...)`.
   * A small Python shim that picks CPU/GPU by config; identical return types.

---

#### Data layout & memory

* **Bins layout**: **column-major** (`[F][N]` contiguous) is preferred to stream a feature across all rows with unit-stride. If your stored tensor is `[N,F]`, create a pinned/NUMA-aware **transposed view** once.
* **Alignment**: 64-byte aligned allocations (`posix_memalign`/`aligned_alloc`).
* **Era index**: store `era_rows[e]` as contiguous `int32` vectors.
* **Quantized grad (optional now, add later)**: store `grad_q[N]` as `int16`, accumulate into `int32`, convert to `float` when computing gains.

---

#### Parallelization plan (OpenMP)

* **Outer**: parallel over *(feature blocks × node blocks)*.
* **Inner**: sequential over eras (so we can do streaming Welford) with SIMD per row.
* Use `#pragma omp parallel for schedule(static)` on the outer loop; pin threads with `proc_bind(close)`; set `OMP_NUM_THREADS=physical_cores`.
* Per thread: local scratch for `counts[B_blk][N_blk][BINS]` and `grads[...]` (avoid false sharing; pad to cache lines).

---

#### SIMD plan (AVX2/AVX-512)

* **Bins** are bytes: load `__m256i` / `__m512i`, compare against 0..4, update **5-bin** hist via masked adds.
* **Grad gather**: prefer **structure of arrays** so grad is a parallel array of rows → unit-stride loads. Use `_mm256_i32gather_ps` only if necessary (it’s slower).
* **Popcount** is not needed; we aggregate per-bin directly.
* **Prefix-sum** on 5 bins is tiny—do scalar or unrolled vector code.

---

#### K-cuts

* **Even**: indices `round(linspace(0, T−1, K))`.
* **Mass**: from per-feature **marginal counts** aggregated over (node, era). Use int64 cumsum + searchsorted; then ensure uniqueness & clamp to `[0, T−1]`.

---

#### DES (streaming Welford across eras)

For each `(feature, node, cut)` keep:

```
wsum, mean, M2         # float32
left_total             # int64 (for min_samples_leaf)
```

For each era `e`:

1. Build **counts** and **grads** hist for that era only.
2. Prefix-sum to get left stats at chosen cuts; right via total−left.
3. Gain per cut: `0.5*(gL^2/(hL+λ) + gR^2/(hR+λ)) − parent_gain`.
4. Weighted update:

```
w_new = wsum + w_e
delta = gain_e - mean
mean += (w_e / w_new) * delta
M2   += w_e * delta * (gain_e - mean)
wsum  = w_new
left_total += left_count_e
```

Optional directional term: accumulate `(agreement, weight)` similarly and add at the end.

---

#### API (C++/pybind11) — skeleton

```cpp
struct SplitDecisionCPU {
  int32_t feature, threshold;
  float   score;
  float   left_grad, right_grad;
  float   left_hess, right_hess;
  int64_t left_count, right_count;
  // optional: vectors of row ids per side per era (if you partition on CPU)
};

std::vector<std::optional<SplitDecisionCPU>>
find_best_splits_batched_cpu(
    span<uint8_t> bins_FxN,     // F-major, contiguous
    span<float>   grad_N,       // or int16
    span<int16_t> era_N,
    const std::vector<NodeShardCPU>& nodes,
    span<int32_t> feature_subset,
    int max_bins, int k_cuts,
    CutSelection cut_sel,       // EVEN or MASS
    float lambda_l2, float lambda_dro, float dir_w,
    int   min_samples_leaf,
    CPUHints hints               // block sizes, threads, simd flags
);
```

**Python binding**: expose as `packboost_cpu.find_best_splits_batched(...)` and return the same schema your Torch path returns.

---

#### Implementation steps (do in order)

1. **Project boilerplate**

   * Add `cpp/` with CMake (C++20), OpenMP, optional AVX-512 flags.
   * `-O3 -march=native -ffast-math -fopenmp`.
   * Set up `pybind11` module `packboost_cpu`.

2. **Memory prep utils**

   * Transpose helper: `[N,F] uint8 → [F][N]` aligned buffer.
   * Era index builder: `std::vector<std::vector<int32_t>> era_rows(E)`.

3. **Reference (non-SIMD) splitter**

   * Single-threaded, clean code that matches Torch math exactly.
   * Unit tests on toy inputs; assert exact parity.

4. **OpenMP parallel version**

   * Parallel over `(feature_block, node_block)`.
   * Era-streaming loop with local hist buffers; Welford accumulators per candidate.

5. **SIMD inner loops**

   * AVX2 path; optional AVX-512 specialization via `#ifdef`.
   * Bench microkernels: (a) hist from bins + grad, (b) prefix-sum + K-gather, (c) gain.

6. **Even & mass K-cuts**

   * Implement both; add tests to ensure cut indices match Python helpers.

7. **Directional term (optional)**

   * Implement only if `direction_weight != 0`.

8. **Row partitioning**

   * Either: return row ids (left/right per era) from CPU, or keep current Torch partitioner. (Pick **one**; fewer copies wins.)

9. **Parity tests**

   * Random seeds, various `K` (0 → full, 4, 8, 16), check:

     * chosen `(feature, threshold)`,
     * `left/right {count, grad, hess}`,
     * `score` within 1e-6–1e-4 tolerance.

10. **Benchmarks**

* Run Synthetic Benchmark - install all required packages (xgboost, lightgbm, catboost).
* Vary threads; report `hist_ms`, `scan_ms`, `score_ms`, total.
* Compare against Torch backend.

---

#### Acceptance criteria

* **Correctness**: parity with Torch path across seeds/configs.
* **Speed**: ≥2× faster than Torch backend at depth 0 on a 32-core server; ≥1.5× at deeper levels (where node counts shrink).
* **Scalability**: good to at least `threads = physical cores`; NUMA regression tests if multi-socket.
* **Stability**: no allocations in hot loops; bounded peak RSS.

---

#### Risks & mitigations

* **NUMA effects**: bind threads to memory; first-touch allocate per-socket.
* **Gather latency**: avoid gather by using F-major bins; if gather unavoidable, block rows to improve L2 reuse.
* **False sharing**: pad hist and Welford accumulators to 64B.

---

#### Deliverables

* `cpp/` library + `packboost_cpu` Python module.
* Unit tests (`pytest`) and parity tests.
* Microbench harness (`bench_cpu.py`) with CSV outputs.
* Short README: build flags, environment, expected speed.

---

#### Tiny code hints (SIMD hist for BINS=5, AVX2)

* Compare against constants 0..4 → five masks; convert masks to `int32` and accumulate.
* Use unrolled loop over bins; maintain `counts[5]` and `grads[5]` in registers; spill once per tile.

---

### M4. CUDA fused frontier (single milestone)

**Goal:** One high-performance CUDA path that subsumes “naive → shared-mem → fused”.

* **Grid/block mapping:** **one warp = one (node, feature)**; blocks tile features; grid tiles nodes.
* **Histograms:** shared-mem histogram `[bins, 2]` for `(∑g, ∑h)` per era using warp reductions; global atomics avoided in the steady state.
* **Scans:** warp shuffles for exclusive prefix over bins; **right = total − left**; optional reverse-cumsum validator in debug builds.
* **K-cuts on-the-fly:**

  * `even`: select K evenly spaced edges (warp-gather).
  * `mass`: compute per-feature CDF from shared hist, select K quantile edges (branch-free search in registers).
* **DES (Welford) in registers:** accumulate mean/std across eras; optional directional term; stable tie-break `(gain ↓, samples ↓, feature ↑, threshold ↑)`.
* **Partition:** segmented scatters into persistent row pools (global offsets precomputed per (node, era, side)).
* **Numerics:** counts `int64`, stats `float32`; bins `uint8/uint16`.
* **Perf gate (A100 or similar):**

  * `2.7M × 2,376 × 5`, `depth=6`, `max_bins=64`, `K=15` ⇒ **≥4×** native CPU path;
  * Numerai-scale throughput target: **\~200 trees/s** at `pack_size=8` (tunable).
* **Correctness gate:** parity vs CPU backend within ≤ **1e−7** predictions and identical split choices under parity settings (`K=full`, validator on).

---

### M5. Precision & memory polish

**Goal:** Reduce bandwidth and footprint without hurting accuracy.

* **Storage:** `uint8` bins whenever `max_bins ≤ 256`; optional **FP16** leaf/score storage with FP32 accumulation.
* **Missing value** support via a dedicated “NaN bin” column or mask lane; predictable routing rule.
* **Optional precomputed lanes (thermometer ≤16):** fast path for small-K workloads (bit-packed masks on GPU), falling back to on-the-fly K-cuts when `max_bins` or K grow.
* **Acceptance:** memory drop ≥ **25%** on reference; prediction parity preserved; speed non-regressing vs M4 within ±5%.

---

### M6. Auto-tuner

**Goal:** Pick good launch parameters automatically.

* **Search space:** `(nodes_tile, features_tile, eras_tile, warps_per_block, rows_per_thread, K_policy)`; device + dataset signature cache.
* **Budget:** ≤ 2–3 seconds warm-up per new signature; reuse thereafter.
* **Acceptance:** **≥15%** end-to-end fit time reduction on 3 varied workloads vs fixed heuristics.

---

### M7. Predictor kernels & serialization

**Goal:** Fast inference and portable bundles.

* **CPU & CUDA predictors:** packed SoA traversal; **tree-block batching**; optional bit-packed thermometer traversal when ≤16 lanes.
* **Serialization:** versioned bundle (trees, feature names, bin edges, config & K-policy) + lightweight loader (`.npz` / `.pt`).
* **Acceptance:** predictor parity vs trainer; inference speedup ≥ **3×** vs Torch router on large batches.

---

### M8. Numerai/Hull integration & benchmarks

**Goal:** Reproducible E2E runs with stability metrics.

* **Pipelines:** training scripts for Numerai and Hull Tactical; DES diagnostics (era mean/std gain, agreement), pack logs (per-depth K, hist/scan/score/partition ms).
* **Metrics:** correlation, sharpe proxy, drawdown control; ablations for `max_bins∈{16,32,64}`, `K∈{7,15,31}`, DES on/off, pack size.
* **Acceptance:** artifact repo with CSVs/plots; improvement stories documented.

---

## QA: frontier & prediction (updated)

* **Parity modes:**

  * *Histogram:* `histogram_mode="rebuild"` (validator reference) and `"subtract"` (fast path).
  * *Thresholds:* `k_cuts=0` → full sweep parity; compare to CPU baseline.
* **Predictor parity:** `predict()` vs `predict_packwise()` → `max_abs_diff ≤ 1e−7`.
* **Accounting invariants:** at each depth, `nodes_subtract_ok + nodes_rebuild == nodes_processed`.
* **Complexity sanity:** with fixed `max_bins`, observed `score_ms ∝ K/(bins−1)` when enabling K-cuts.
* **Perf microbench:** `200k × 64 feats, depth 6` → log `hist_ms, scan_ms, score_ms, partition_ms`; expect ≥ **20%** drop in `hist_ms + scan_ms` vs pre-M1 Python frontier; ≥ **2×** vs Torch on M2; ≥ **4×** vs M2 on M4 GPU.

---

## 10) Changelog

- 2025-09-23 — M1.2 / M2.1 alignment: prebinned path, DES parity, hull benchmark polish
  - Added `PackBoostConfig.prebinned` for workflows that supply integer bins; `fit(..., era=None)` now defaults to a single era (useful for DES-off benchmarking).
  - Torch and native CPU frontiers collapse duplicate node shards (root histogram reuse) so packs share histogram/scan work; instrumentation reports `nodes_collapsed` for visibility.
  - DES directional term now matches the CUDA `directional_split_kernel`: per-era direction = `sign(left_value - right_value)` with weighted averaging, eliminating the old parent-direction heuristic.
  - `examples/hull_benchmark.py` tightened preprocessing (explicit DES-off path, optional prebinned flag, no-leak era masks) to document the M1/M2 training flow.

- 2025-09-21 — M1.1: Subtract-First Frontier + Packed Prediction
  - Refactored `_find_best_splits_batched`:
    - One-time concat of (rows, era_ids, node_ids, grad, hess) per depth.
    - Histogram counts remain **int64** end-to-end.
    - **Subtract-first** policy in `auto`: right = total − left; reverse-cumsum rebuild only if validator fails.
    - Single era-wise prefix source; aggregated prefixes via `sum(dim=era)`.
  - Prediction:
    - Replaced Python DFS with **vectorized per-row routing** (flat SoA cache per device).
    - Added `predict_packwise(block_size_trees=16)` for faster inference with exact parity.
  - Determinism & QA:
    - Stable tiebreak `(gain ↓, sample_count ↓, feature_id ↑, threshold ↑)`.
    - Depth logs assert `nodes_subtract_ok + nodes_rebuild == nodes_processed`.
  - Perf (200k×64 @ depth 6): hist_ms=11688.90, scan_ms=171.83, score_ms=632.51, partition_ms=362.26 (hist+scan=11860.72 ms).
- 2025-09-22 — Locked M1 design guardrails: enabled node-batched frontier with
  pack-average updates, configurable histogram policies (`rebuild`/`subtract`/`auto`),
  per-depth instrumentation, and regression tests covering pack weighting,
  histogram subtraction invariants, and batching parity.
- 2025-09-21 — Implemented batched histogram builder and vectorized split scoring for faster Milestone 1 training.
- 2025-09-20 — Applied Milestone 1 audit fixes: global parent direction in DES, weighted Welford with `era_alpha`, histogram subtraction guard, DES regression tests, native backend stubs, and CPU-only documentation updates.
