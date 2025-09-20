# PackBoost Project Compass

## 1. Problem Statement & Theory Overview
PackBoost targets fast gradient-boosted trees tailored to the Numerai workflow where the objective blends correlation, stability across eras, and risk control. Each boosting round fits a *pack* of trees jointly, sharing the sampled features at each depth and optimising a **Directional Era Splitting (DES)** score per node.

- **Boosting objective.** At round \(t\), predictions are updated \(\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \sum_{b=1}^B f_{t,b}(x)\), with pack size \(B\) and learning rate \(\eta\).
- **Node score** (*per feature-threshold candidate*):
  \[
  \text{score} = \underbrace{\mu_G}_{\text{mean gain}} - \lambda_{\text{dro}}\, \underbrace{\sigma_G}_{\text{per-era std dev}} + \lambda_{\text{dir}}\, \text{directional agreement}
  \]
  Gains are computed from gradients \(g_i\) and hessians \(h_i\) aggregated by era; agreement counts eras where left/right leaf predictions match the global direction.
- **Constraints.** Minimum samples per leaf, optional prebinned inputs, shared feature subsampling per depth.
- **GPU pathway.** Frontier evaluation is translated into histogram building on device, using per-era partitioning and thrust-based reductions to avoid host round-trips.

## 2. Mathematical Formulation
For a candidate split (feature \(k\), threshold \(\tau\)) on node sample set \(S\) with era partition \(S_e\):

- Gradients/Hessians per child:
  \[G_L = \sum_{i \in S_L} g_i, \quad H_L = \sum_{i \in S_L} h_i, \quad G_R = G_T - G_L\]
- Gains (with L2 regularisation \(\lambda\)):
  \[\text{gain}_L = \frac{1}{2}\frac{G_L^2}{H_L + \lambda},
    \quad \text{gain}_R = \frac{1}{2}\frac{G_R^2}{H_R + \lambda}
  \]
- Parent gain \(\text{gain}_T\) computed analogously.
- Per-era gains used to compute \(\mu_G, \sigma_G\). Missing eras are treated as zero-contribution but counted towards agreement neutrality.
- Leaf predictions: \(v_L = -\frac{G_L}{H_L + \lambda}, v_R = -\frac{G_R}{H_R + \lambda}\).

## 3. Design Choices
### Current
- **Pack boosting** to parallelise tree growth and share feature subsets per depth.
- **DES scoring** emphasising stable, directionally aligned splits across eras.
- **Host/GPU split:** CPU reference implementation kept for correctness; CUDA workspace introduced to persist binned matrices, gradients, and metadata on device.
- **Quantile binning** with optional prebinned shortcut when input already consists of indices.
- **Config-driven** architecture via `PackBoostConfig` for reproducibility and portability.

### Future Considerations
1. Mixed precision storage (e.g. uint16 bins) to trim device memory.
2. Async data transfers / overlapping kernel execution for streaming batches.
3. Enhanced regularisation (e.g. gradient clipping, monotonic constraints, adversarial robustness).
4. Sparse feature support and missing value handling directly in kernels.
5. Automated hyper-parameter search tuned for Numerai objectives.
6. Multi-GPU sharding across packs for >200 tree/s throughput goals.

## 4. Blueprint Alignment
ExtraFastBooster remains the design compass. It grows an entire pack of trees layer-by-layer, samples a common feature slice per depth, and encodes each level of every tree into compact arrays. Before histogramming, those encodings are rearranged so that feature-major access is contiguous, letting the GPU kernels `_prep_vars`, `_repack_trees_for_features`, and `_unweighted_featureless_histogram` stride through the frontier with minimal divergence. The offline predictor follows the same depth-wise packed layout, advancing leaf slots and summing logits without synchronising with the host. PackBoost inherits the orchestration, DES-first scoring, and histogram flow while replacing the exploratory notebooks with production-grade Python plus native backends that understand DES-era tiling.

## 5. Target Architecture
**Python package layout**

- `packboost/config.py`: frozen dataclasses covering pack size, depth, DES regularisation, tiling knobs, device choice, RNG seed, and related hyperparameters.
- `packboost/data.py`: preprocessing toolkit that detects prebinned matrices, performs quantile binning into `uint8` buckets when needed, groups rows by era, samples features, and provides gradient/hessian materialisation.
- `packboost/core.py`: public PackBoost estimator that coordinates boosting rounds and mirrors the interface used in the README, benchmarks, and notebooks so existing entry points continue to function.
- `packboost/frontier.py`: dispatcher choosing between CPU and CUDA frontiers, marshalling numpy/cupy buffers, and returning split decisions.
- `packboost/workspace.py`: persistent scratch buffers for histograms, prefix scans, and per-era temporaries—mirroring ExtraFastBooster’s GPU workspaces but with maintainable host/device abstractions.
- `packboost/predict.py`: shared inference utilities for both CPU and GPU plus a serialisable predictor class for quick-start usage.

**Native extension layout**

- `packboost/backends/cpu/frontier.cpp`: OpenMP + AVX era-tiled DES evaluator.
- `packboost/backends/cuda/frontier.cu`: CUDA frontier kernel with cooperative group synchronisation and shared-memory histograms.
- `packboost/backends/bindings.cpp`: pybind11 surface exposing the CPU and CUDA entry points along with helper routines for inference.

Build integration remains in `setup_native.py`, controlled by environment variables as described in the README so CPU-only users can opt out cleanly.

## 6. Training Round Walkthrough
1. **Root preparation**: Detect prebinned inputs; otherwise quantile-bin once on the CPU (no bit-packing). Build an `EraShardIndex` that records row indices per era per node starting at the root. Allocate prediction, gradient, and hessian buffers as `float32`.
2. **Depth loop**: For each level, sample a common feature subset for the entire pack, partition active nodes into `(nodes_tile × features_tile)` blocks sized to respect workspace memory limits, and iterate tile-by-tile.
3. **Backend invocation**: Each tile call receives node-era views, the feature-major bin matrix (`uint16`), gradient/hessian arrays, and configuration (max bins, L2 lambda, DES lambda, era tile size).
4. **Backend responsibilities**:
   - Tile eras to manageable chunks and build `[nodes_tile, features_tile, bins, eras, 2]` histograms in shared buffers.
   - Prefix-sum histograms to get left/right aggregates per threshold.
   - For every era, compute Newton gains and feed them into Welford accumulators tracking mean, variance, and directional agreement.
   - After processing all era tiles, collapse scores to obtain the DES objective (mean − λ·std ± directional penalty) and return the best `(feature, threshold, gain, stats)` per node.
   - Produce child partitions by performing stable partitions on the CPU or scatter/gather on the GPU without synchronising to the host.
5. **Python updates**: The orchestrator records split details, caches child era shards, computes leaf values `-G/(H+λ)`, applies the learning rate, updates predictions, and stores compressed structure-of-arrays representations for inference.

## 7. CPU Backend Blueprint (`frontier.cpp`)
- **Layout**: Consume contiguous `uint16` bins, `float32` grads/hessians, and `int32` row indices.
- **Tiling**: Use nested OpenMP `parallel for` loops over node and feature tiles; stream eras within inner loops.
- **Histogramming**: Prefer AVX-512 gathers when present, otherwise fall back to manual unrolled loops; keep histograms in `float32` and flush after each era tile.
- **DES aggregation**: Implement an aligned struct storing count, mean, M2, and agreement counters to run Welford updates efficiently.
- **Partitioning**: Execute stable partitions per era with prefix sums to compute offsets and write child row indices into reusable buffers exposed to Python via capsules or workspace handles.
- **Threading**: Provide thread-local histograms and reduce them to avoid locks; the entire path stays synchronous.

## 8. CUDA Backend Blueprint (`frontier.cu`)
- **Scheduling**: Launch grids over node and feature tiles; within blocks, iterate era tiles while coordinating threads via cooperative groups or `cuda::std::barrier`.
- **Histogram kernel**: Load row indices into shared memory chunks, accumulate `[bins, 2]` grad/hess stats with warp reductions, and reuse shared memory across era tiles.
- **Prefix and scoring**: Perform warp-level exclusive scans to compute cumulative sums, evaluate Newton gains, and update Welford statistics kept in registers.
- **Directional agreement**: Track the sign of child predictions per era inline, updating agreement counters without extra passes.
- **Output**: Store best splits in shared memory and commit once per block. Build child partitions through segmented scatters using warp scans and write to device workspaces. Return compact descriptors to Python using async copies or stream-ordered events—never `cudaDeviceSynchronize`.
- **Memory reuse**: Maintain `CudaFrontierWorkspace` with persistent row-index pools, histogram scratch, and Welford buffers sized from configuration and reused every depth.

## 9. Python Orchestrator Responsibilities
`PackBoost.fit` keeps a deterministic RNG for feature and row sampling, delegates frontier evaluation to the backend, vectorises prediction updates, and surfaces hooks for callbacks, logging, and optional early stopping. `PackBoost.save/load` serialises tree packs and metadata, while `PackBoost.predict` reuses the packed arrays to traverse trees depth-by-depth on CPU or via optional CUDA kernels.

## 10. Throughput Targets and Optimisations
- Aim for ≈200 trees/second on the Numerai workload (2.7M rows × 2,376 features × 5 bins on A100).
- Use `uint16` bins with `float32` stats to balance bandwidth and precision.
- Size era tiles (e.g., 64) to keep shared-memory usage within hardware limits (≈96 KB).
- Allow configurable launch geometry (threads/block, rows/thread) with future auto-tuning.
- Exploit `cp.async` where available to overlap global memory reads with computation.
- Keep Welford accumulators in registers and flush once per `(node, feature)`.
- Build partitions via warp scans or CUB primitives to avoid divergence.
- Pin gradient/hessian memory for asynchronous updates and rely on stream events rather than global device syncs.

## 11. Testing & Benchmarking Strategy
- **Unit tests**: DES/Welford correctness against NumPy baselines, partition stability per era, CPU vs GPU prediction parity, configuration serialisation round-trips.
- **Integration tests**: Exercise `examples/synthetic_benchmark.py` and a trimmed Numerai pipeline to ensure API compatibility and reproducible outputs.
- **Performance harness**: Provide `benchmarks/numerai_throughput.py` to measure trees/s on pre-binned Numerai samples, logging that the implementation meets or exceeds the throughput goal on target hardware.

## 12. Milestone Roadmap
1. Scaffold the Python package, configuration system, and pure Python pack booster that mirrors ExtraFastBooster logic without native dependencies.
2. Deliver the CPU frontier backend with DES-era tiling and validate it against the Python baseline through tests.
3. Implement the CUDA frontier kernel to match the CPU algorithm, ensuring deterministic outputs and independence from device-wide synchronisation.
4. Optimise memory layouts and kernel behaviour, then benchmark on Numerai-scale data to hit the throughput target.
5. Finalise documentation, notebooks, and examples to reflect the new PackBoost pipeline.

This compass should be refreshed whenever the architecture or algorithm evolves; treat `README.md` as the user-facing quickstart, while `project.md` remains the engineering blueprint.
