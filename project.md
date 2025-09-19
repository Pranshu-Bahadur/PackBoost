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

## 4. Implementation Steps (current status)
1. **Repository scaffolding**: pure Python booster with CPU fallback.
2. **CUDA backend**: histogram and frontier kernels, pybind11 bindings (`packboost/backends/src`).
3. **Workspace abstraction**: `CudaFrontierWorkspace` persists device buffers and rebuilds node/era groupings with thrust.
4. **Booster integration**: GPU path leverages workspace when available; gradients/hessians pinned for reuse.
5. **Packaging**: `setup_native.py` handles optional CUDA compile, `setup.py` re-exports build helper, and `pyproject.toml` supplies build requirements.
6. **Testing/CI**: pytest suite covers DES scoring equivalence and sklearn wrapper.

## 5. Workflow Log (recent milestones)
- **Mar 2025**: Introduced CUDA workspace, device-resident frontier evaluation, and python-side integration.
- **Mar 2025**: Packaging overhaul; editable install builds native backend, optional CUDA.
- **Mar 2025**: Debugged namespace linkage to expose CUDA bindings; added runtime diagnostics for backend load failures.

## 6. Current Optimisations
- Device-side concatenation of node samples and era grouping using thrust reduces CPU bottleneck.
- Gradients/hessians reused in-place (`np.subtract` + pinned registration) to avoid allocations each depth.
- Shared feature subset across pack to reduce histogram builds.
- Configurable kernel launch parameters (`threads_per_block`, `rows_per_thread`).
- Prebin detection (integers in range) skips quantile binning.

## 7. Future Work Plan
1. **Performance Profiling**: integrate Nsight/torch.profiler traces to target hotspots (shared memory limits, atomic contention).
2. **Memory Optimisation**: compress indices, reuse temporary device vectors, adopt pooling allocator.
3. **Era-aware batching**: dynamic tiling sized to warp occupancy.
4. **Model features**: support monotonic constraints, categorical splits, early stopping, custom objective interfaces.
5. **User experience**: CLI/Notebook utilities, richer logging, comparison notebooks vs CatBoost/LightGBM.
6. **Benchmarking**: automated runs on Numerai dataset, track depth-7 throughput vs target (200 trees/s).

This document should be updated alongside significant architectural or algorithmic changes; treat `README.md` as user-facing quickstart, while `project.md` remains the engineering compass.
