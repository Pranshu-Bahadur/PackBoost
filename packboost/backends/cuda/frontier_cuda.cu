// frontier_cuda.cu  (feature-major bins; 64-bit safe; barrier-safe scan; block-wide totals)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace py::literals;

namespace {

#define CUDA_CHECK(expr) do { cudaError_t e__=(expr); if(e__!=cudaSuccess){ \
  throw std::runtime_error(std::string("CUDA error: ")+cudaGetErrorString(e__)); }} while(0)

constexpr int WARP_SIZE = 32;
constexpr int MAX_BINS  = 128;
constexpr int THREADS   = 512;                  // 8 warps/block is a good default
constexpr int WARPS     = THREADS / WARP_SIZE;
__device__ constexpr float NEG_INF = -1.0e30f;

__host__ __device__ inline int max_int(int a,int b){return a>b?a:b;}
__host__ __device__ inline int min_int(int a,int b){return a<b?a:b;}
__host__ __device__ inline int clamp_int(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}

__device__ inline float warp_reduce_sum(float v){
  for(int o=WARP_SIZE/2;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
  return v;
}
__device__ inline int warp_reduce_sum_int(int v){
  for(int o=WARP_SIZE/2;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
  return v;
}

// Feature-major accessor: bins on device are [F, N]
__device__ __forceinline__ int load_bin_colmajor(const uint8_t* bins, int N, int row, int feat){
  return int(bins[(size_t)feat * (size_t)N + (size_t)row]);
}

// Safe Hillis–Steele inclusive scan on first n entries of g/h/c
__device__ inline void block_scan_prefix(float* g, float* h, int* c, int n) {
  for (int off = 1; off < n; off <<= 1) {
    __syncthreads();
    int b = threadIdx.x;
    float gp = 0.f, hp = 0.f; int cp = 0;
    if (b < n && b >= off) { gp = g[b - off]; hp = h[b - off]; cp = c[b - off]; }
    __syncthreads();
    if (b < n) { g[b] += gp; h[b] += hp; c[b] += cp; }
  }
  __syncthreads();
}

// ===== Batched frontier partition: count + warp-aggregated scatter =====

template<int TPB>
__global__ void frontier_count_kernel(
    const int8_t*  __restrict__ bins_colmajor,   // [F,N] feature-major int8
    int            rows_dataset,                 // N
    const int32_t* __restrict__ rows_index,      // [Rcat] -> dataset row ids
    const int32_t* __restrict__ node_row_splits, // [Nsel+1] offsets into rows_index
    const int32_t* __restrict__ node_era_splits, // [Nsel*(E+1)] offsets within node span
    const int32_t* __restrict__ feat_per_node,   // [Nsel]
    const int32_t* __restrict__ thr_per_node,    // [Nsel]
    int32_t*       __restrict__ left_counts,     // [Nsel*E] out
    int            Nsel,
    int            E
){
    const int n = blockIdx.x;                    // node
    if (n >= Nsel) return;

    const int feat = feat_per_node[n];
    const int thr  = thr_per_node[n];

    const int32_t node_base = node_row_splits[n];
    const int32_t* era_off  = node_era_splits + n * (E + 1);

    for (int e = 0; e < E; ++e) {
        const int32_t beg = era_off[e];
        const int32_t end = era_off[e + 1];
        const int32_t len = end - beg;

        // per-thread local counter
        int local = 0;
        for (int i = threadIdx.x; i < len; i += TPB) {
            const int32_t pos  =  beg + i;
            const int32_t rid  = rows_index[pos];
            const int bin      = load_bin_colmajor(reinterpret_cast<const uint8_t*>(bins_colmajor), rows_dataset, rid, feat);
            local += (bin <= thr);
        }

        // block reduction
        __shared__ int ssum[TPB];
        ssum[threadIdx.x] = local;
        __syncthreads();
        for (int stride = TPB >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) ssum[threadIdx.x] += ssum[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) left_counts[n * E + e] = ssum[0];
        __syncthreads();
    }
}

__device__ __forceinline__ unsigned lane_id_u() { return threadIdx.x & (WARP_SIZE - 1); }

// Each block handles one node; per era we stream rows and do warp-aggregated atomics.
template<int TPB>
__global__ void frontier_scatter_kernel(
    const int8_t*  __restrict__ bins_colmajor,   // [F,N]
    int            rows_dataset,                 // N
    const int32_t* __restrict__ rows_index,      // [Rcat]
    const int32_t* __restrict__ node_row_splits, // [Nsel+1]
    const int32_t* __restrict__ node_era_splits, // [Nsel*(E+1)]
    const int32_t* __restrict__ feat_per_node,   // [Nsel]
    const int32_t* __restrict__ thr_per_node,    // [Nsel]
    const int32_t* __restrict__ left_splits,     // [Nsel*(E+1)] exclusive
    const int32_t* __restrict__ right_splits,    // [Nsel*(E+1)] exclusive
    int32_t*       __restrict__ out_left,        // [sum_left]
    int32_t*       __restrict__ out_right,       // [sum_right]
    int            Nsel,
    int            E
){
    const int n = blockIdx.x;                    // node
    if (n >= Nsel) return;

    const int feat = feat_per_node[n];
    const int thr  = thr_per_node[n];

    const int32_t node_base = node_row_splits[n];
    const int32_t* era_off  = node_era_splits + n * (E + 1);
    const int32_t* lsplit   = left_splits     + n * (E + 1);
    const int32_t* rsplit   = right_splits    + n * (E + 1);

    extern __shared__ unsigned char smem[];
    int32_t* curL = reinterpret_cast<int32_t*>(smem);          // [E]
    int32_t* curR = reinterpret_cast<int32_t*>(smem) + E;      // [E]

    // init cursors
    for (int e = threadIdx.x; e < E; e += TPB) {
        curL[e] = 0;
        curR[e] = 0;
    }
    __syncthreads();

    for (int e = 0; e < E; ++e) {
        const int32_t beg = era_off[e];
        const int32_t end = era_off[e + 1];
        const int32_t len = end - beg;

        for (int i = threadIdx.x; i < len; i += TPB) {
            const bool active = (i < len);
            int32_t rid = -1, bin = 0;
            if (active) {
                const int32_t pos  = beg + i;
                rid = rows_index[pos];
                bin = load_bin_colmajor(reinterpret_cast<const uint8_t*>(bins_colmajor), rows_dataset, rid, feat);
            }
            const int is_left  = active && (bin <= thr);
            const int is_right = active && !is_left;

            // LEFT: warp-aggregated
            const unsigned warp_mask = __ballot_sync(0xffffffff, is_left);
            const int n_left = __popc(warp_mask);
            const unsigned lane = lane_id_u();
            int warp_baseL = 0;
            if (lane == 0 && n_left > 0) warp_baseL = atomicAdd(&curL[e], n_left);
            warp_baseL = __shfl_sync(0xffffffff, warp_baseL, 0);
            if (is_left) {
                const int rank = __popc(warp_mask & ((1u << lane) - 1));
                out_left[lsplit[e] + warp_baseL + rank] = rid;
            }

            // RIGHT: warp-aggregated
            const unsigned warp_mask_r = __ballot_sync(0xffffffff, is_right);
            const int n_right = __popc(warp_mask_r);
            int warp_baseR = 0;
            if (lane == 0 && n_right > 0) warp_baseR = atomicAdd(&curR[e], n_right);
            warp_baseR = __shfl_sync(0xffffffff, warp_baseR, 0);
            if (is_right) {
                const int rank = __popc(warp_mask_r & ((1u << lane) - 1));
                out_right[rsplit[e] + warp_baseR + rank] = rid;
            }
        }
        __syncthreads();
    }
}


__global__ void cuda_predict_bins_kernel(
    const uint8_t* __restrict__ bins_colmajor,  // [F, N] (feature-major)
    int num_rows,

    const int32_t* __restrict__ feature,        // [num_nodes]
    const int32_t* __restrict__ threshold,      // [num_nodes]
    const int32_t* __restrict__ left,           // [num_nodes]
    const int32_t* __restrict__ right,          // [num_nodes]
    const uint8_t* __restrict__ is_leaf,        // [num_nodes] (0/1)
    const float*   __restrict__ value,          // [num_nodes]
    int num_nodes,

    float* __restrict__ out                     // [N]
){
    // grid-stride: one thread routes many rows independently
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < num_rows;
         row += blockDim.x * gridDim.x)
    {
        int node = 0;
        // Safety cap to avoid rare malformed trees causing infinite loops
        // (depth <= num_nodes for well-formed binary trees)
        for (int it = 0; it < num_nodes; ++it) {
            if (is_leaf[node]) {
                out[row] = value[node];
                break;
            }
            const int f   = feature[node];
            const int thr = threshold[node];
            const int v   = load_bin_colmajor(bins_colmajor, num_rows, row, f);
            const bool go_left = (v <= thr);
            node = go_left ? left[node] : right[node];

            // Defensive guard: if nodes are corrupted, emit 0 and stop
            if (node < 0 || node >= num_nodes) {
                out[row] = 0.0f;
                break;
            }
        }
    }
}

__global__ void cuda_find_best_splits_kernel(
  // bins is FEATURE-MAJOR: [num_features_dataset, rows_dataset]
  const int8_t* __restrict__ bins_fm,
  int rows_dataset,                                 // = number of rows in dataset (N)
  int cols_dataset,                                 // = number of features in dataset (F)
  const float* __restrict__ grad,                  // [rows_total_compact]
  const float* __restrict__ hess,                  // [rows_total_compact]
  const int32_t* __restrict__ rows_index,          // [rows_total_compact] -> original dataset row ids
  const int32_t* __restrict__ node_row_splits,     // [num_nodes+1] (offsets into grad/hess/rows_index)
  const int32_t* __restrict__ node_era_splits,     // [num_nodes*(num_eras+1)] (same indexing space)
  const float*   __restrict__ era_weights,         // [num_nodes*num_eras]
  const float*   __restrict__ total_grad_nodes,    // [num_nodes]
  const float*   __restrict__ total_hess_nodes,    // [num_nodes]
  const int64_t* __restrict__ total_count_nodes,   // [num_nodes]
  const int32_t* __restrict__ feature_ids,         // [num_features]
  int num_nodes,
  int num_features,
  int num_bins,
  int num_eras,
  int k_cuts,
  int cut_mode,                                    // 0 even, 1 mass
  int min_samples_leaf,
  float lambda_l2,
  float lambda_dro,
  float direction_weight,
  int rows_total_compact,                          // == len(rows_index)
  // outputs
  float*   __restrict__ out_scores,                // [num_nodes*num_features]
  int32_t* __restrict__ out_thresholds,            // [num_nodes*num_features]
  float*   __restrict__ out_left_grad,             // [num_nodes*num_features]
  float*   __restrict__ out_left_hess,             // [num_nodes*num_features]
  int64_t* __restrict__ out_left_count             // [num_nodes*num_features]
){
  const int node_id    = blockIdx.y;
  const int foff       = blockIdx.x;
  const int lane       = threadIdx.x & (WARP_SIZE-1);
  const int warp_id    = threadIdx.x / WARP_SIZE;
  __shared__ int s_Keval;

  if(node_id>=num_nodes || foff>=num_features) return;

  const int feature_id = feature_ids[foff];
  const int out_index  = node_id * num_features + foff;

  if(feature_id<0 || feature_id>=cols_dataset || num_bins<=1){
    if(threadIdx.x==0){
      out_scores[out_index]=NEG_INF; out_thresholds[out_index]=-1;
      out_left_grad[out_index]=0.f; out_left_hess[out_index]=0.f; out_left_count[out_index]=0;
    } return;
  }

  const int32_t* era_off = node_era_splits + node_id*(num_eras+1);

  int row_start=node_row_splits[node_id];
  int row_end  =node_row_splits[node_id+1];
  row_start=max_int(0,row_start);
  row_end  =min_int(rows_total_compact,row_end);
  if(row_start>=row_end){
    if(threadIdx.x==0){
      out_scores[out_index]=NEG_INF; out_thresholds[out_index]=-1;
      out_left_grad[out_index]=0.f; out_left_hess[out_index]=0.f; out_left_count[out_index]=0;
    } return;
  }
  if(row_end-row_start < 2*min_samples_leaf){
    if(threadIdx.x==0){
      out_scores[out_index]=NEG_INF; out_thresholds[out_index]=-1;
      out_left_grad[out_index]=0.f; out_left_hess[out_index]=0.f; out_left_count[out_index]=0;
    } return;
  }

  // ---- Shared memory ----
  extern __shared__ unsigned char smem[];
  unsigned char* cur=smem;

  // per-warp private histograms
  int32_t* cnt_w = reinterpret_cast<int32_t*>(cur); cur += WARPS*num_bins*sizeof(int32_t);
  float*   grd_w = reinterpret_cast<float*>(cur);   cur += WARPS*num_bins*sizeof(float);
  float*   hss_w = reinterpret_cast<float*>(cur);   cur += WARPS*num_bins*sizeof(float);

  // reduced hist
  int32_t* count_bins = reinterpret_cast<int32_t*>(cur); cur += num_bins*sizeof(int32_t);
  float*   grad_bins  = reinterpret_cast<float*>(cur);   cur += num_bins*sizeof(float);
  float*   hess_bins  = reinterpret_cast<float*>(cur);   cur += num_bins*sizeof(float);

  // prepass total (for mass cuts)
  int32_t* count_total = reinterpret_cast<int32_t*>(cur); cur += num_bins*sizeof(int32_t);

  // thresholds
  int32_t* thr_sh = reinterpret_cast<int32_t*>(cur); cur += num_bins*sizeof(int32_t);

  // per-threshold accumulators
  float* mean_arr      = reinterpret_cast<float*>(cur); cur += num_bins*sizeof(float);
  float* M2_arr        = reinterpret_cast<float*>(cur); cur += num_bins*sizeof(float);
  float* wsum_arr      = reinterpret_cast<float*>(cur); cur += num_bins*sizeof(float);
  float* dir_arr       = reinterpret_cast<float*>(cur); cur += num_bins*sizeof(float);
  float* leftg_arr     = reinterpret_cast<float*>(cur); cur += num_bins*sizeof(float);
  float* lefth_arr     = reinterpret_cast<float*>(cur); cur += num_bins*sizeof(float);
  // align for int64
  uintptr_t ap = (reinterpret_cast<uintptr_t>(cur) + alignof(int64_t)-1) & ~(alignof(int64_t)-1);
  int64_t* leftc_arr   = reinterpret_cast<int64_t*>(ap);
  cur = reinterpret_cast<unsigned char*>(leftc_arr + num_bins);

  // per-warp partial totals → block-wide reduce
  float* s_tg = reinterpret_cast<float*>(cur); cur += WARPS * sizeof(float);
  float* s_th = reinterpret_cast<float*>(cur); cur += WARPS * sizeof(float);
  int*   s_tc = reinterpret_cast<int*>(cur);   cur += WARPS * sizeof(int);
  // shared broadcast slots
  float* s_era_weight  = reinterpret_cast<float*>(cur); cur += sizeof(float);
  float* s_parent_gain = reinterpret_cast<float*>(cur); cur += sizeof(float);

  const int Tfull = max_int(num_bins-1,1);
  int Keval = 0;

  // zero accumulators
  for(int i=threadIdx.x;i<num_bins;i+=blockDim.x){
    mean_arr[i]=0.f; M2_arr[i]=0.f; wsum_arr[i]=0.f; dir_arr[i]=0.f;
    leftg_arr[i]=0.f; lefth_arr[i]=0.f; leftc_arr[i]=0;
    count_total[i]=0;
  }
  __syncthreads();

  // ---- MASS pre-pass (counts across eras) to select thresholds ----
  if(k_cuts>0 && k_cuts<Tfull && cut_mode==1){
    for(int b=(threadIdx.x & (WARP_SIZE-1)); b<num_bins; b+=WARP_SIZE) cnt_w[warp_id*num_bins+b]=0;
    __syncthreads();

    for(int r=row_start+threadIdx.x; r<row_end; r+=blockDim.x){
      const int32_t ridx = rows_index[r];
      const int bin = load_bin_colmajor(reinterpret_cast<const uint8_t*>(bins_fm), rows_dataset, ridx, feature_id);
      if(bin>=0 && bin<num_bins) atomicAdd(&cnt_w[warp_id*num_bins+bin],1);
    }
    __syncthreads();

    for(int b=threadIdx.x;b<num_bins;b+=blockDim.x){
      int s=0; 
      #pragma unroll
      for(int w=0;w<WARPS;++w) s += cnt_w[w*num_bins+b];
      count_total[b]=s;
    }
    __syncthreads();

    if(threadIdx.x==0){
      const int K = k_cuts;
      if(K<=0 || K>=Tfull){ Keval=Tfull; for(int t=0;t<Keval;++t) thr_sh[t]=t; }
      else{
        long long total=0; for(int b=0;b<num_bins;++b) total += (long long)count_total[b];
        if(total<=0){
          Keval = K;
          if(K==1){ thr_sh[0]=0; }
          else{
            double step=double(Tfull-1)/double(K-1);
            for(int t=0;t<K;++t){ int th=int(std::round(step*t)); thr_sh[t]=clamp_int(th,0,Tfull-1); }
          }
        }else{
          int cand=0; Keval=K;
          for(int i=0;i<K;++i){
            double a = (K==1)?0.0: double(i)/double(K-1);
            double tgt = double(total)*(1.0-1e-12)*a;
            long long run=0; int sel=0;
            for(int b=0;b<num_bins;++b){ run += (long long)count_total[b]; if(run>=tgt){ sel=b; break; } }
            thr_sh[cand++] = clamp_int(sel-1,0,Tfull-1);
          }
          // sort-unique
          for(int i=1;i<cand;++i){ int key=thr_sh[i], j=i-1; while(j>=0 && thr_sh[j]>key){ thr_sh[j+1]=thr_sh[j]; --j;} thr_sh[j+1]=key; }
          int uniq=0; for(int i=0;i<cand;++i) if(i==0 || thr_sh[i]!=thr_sh[i-1]) thr_sh[uniq++]=thr_sh[i];
          while(uniq<K) { thr_sh[uniq]=thr_sh[uniq-1]; ++uniq; }
          Keval = min_int(uniq,K);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) s_Keval = Keval;
    __syncthreads();
    Keval = s_Keval;
  } else {
    if(threadIdx.x==0){
      if(k_cuts<=0 || k_cuts>=Tfull){ Keval=Tfull; for(int t=0;t<Keval;++t) thr_sh[t]=t; }
      else{
        Keval=k_cuts;
        if(k_cuts==1) thr_sh[0]=0;
        else{
          double step=double(Tfull-1)/double(k_cuts-1);
          for(int t=0;t<k_cuts;++t){ int th=int(std::round(step*t)); thr_sh[t]=clamp_int(th,0,Tfull-1); }
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) s_Keval = Keval;
    __syncthreads();
    Keval = s_Keval;
  }

  Keval = min_int(Keval, MAX_BINS-1);
  if(Keval<=0){
    if(threadIdx.x==0){
      out_scores[out_index]=NEG_INF; out_thresholds[out_index]=-1;
      out_left_grad[out_index]=0.f; out_left_hess[out_index]=0.f; out_left_count[out_index]=0;
    } return;
  }

  // ---- Main per-era loop (build hist -> scan -> DES update) ----
  for(int era=0; era<num_eras; ++era){
    int es = max_int(era_off[era],   row_start);
    int ee = min_int(era_off[era+1], row_end);
    if(es>=ee) continue;

    // zero per-warp privates
    for(int b=(threadIdx.x & (WARP_SIZE-1)); b<num_bins; b+=WARP_SIZE){
      cnt_w[warp_id*num_bins+b]=0; grd_w[warp_id*num_bins+b]=0.f; hss_w[warp_id*num_bins+b]=0.f;
    }
    __syncthreads();

    // build per-warp histograms
    for(int r=es+threadIdx.x; r<ee; r+=blockDim.x){
      const int32_t ridx = rows_index[r];
      const int bin = load_bin_colmajor(reinterpret_cast<const uint8_t*>(bins_fm), rows_dataset, ridx, feature_id);
      if(bin>=0 && bin<num_bins){
        const float g=grad[r], h=hess[r];
        atomicAdd(&cnt_w[warp_id*num_bins+bin],1);
        atomicAdd(&grd_w[warp_id*num_bins+bin],g);
        atomicAdd(&hss_w[warp_id*num_bins+bin],h);
      }
    }
    __syncthreads();

    // reduce warps -> block histogram
    for(int b=threadIdx.x; b<num_bins; b+=blockDim.x){
      int   c=0; float g=0.f, hh=0.f;
      #pragma unroll
      for(int w=0; w<WARPS; ++w){ c+=cnt_w[w*num_bins+b]; g+=grd_w[w*num_bins+b]; hh+=hss_w[w*num_bins+b]; }
      count_bins[b]=c; grad_bins[b]=g; hess_bins[b]=hh;
    }
    __syncthreads();

    // ---- totals for this era (block-wide) ----
    float tg_part = 0.f, th_part = 0.f; int tc_part = 0;
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
      tg_part += grad_bins[b]; th_part += hess_bins[b]; tc_part += count_bins[b];
    }
    tg_part = warp_reduce_sum(tg_part);
    th_part = warp_reduce_sum(th_part);
    tc_part = warp_reduce_sum_int(tc_part);
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
      const int w = threadIdx.x / WARP_SIZE;
      s_tg[w] = tg_part; s_th[w] = th_part; s_tc[w] = tc_part;
    }
    __syncthreads();
    float total_grad_e = 0.f, total_hess_e = 0.f; int total_count_e = 0;
    if (threadIdx.x == 0) {
      for (int w = 0; w < WARPS; ++w) { total_grad_e += s_tg[w]; total_hess_e += s_th[w]; total_count_e += s_tc[w]; }
      *s_era_weight  = era_weights[node_id * num_eras + era];
      *s_parent_gain = 0.5f * (total_grad_e * total_grad_e) / (total_hess_e + lambda_l2);
      // stash block totals in the first warp slots for broadcast if needed
      s_tg[0] = total_grad_e; s_th[0] = total_hess_e; s_tc[0] = total_count_e;
    }
    __syncthreads();
    total_grad_e  = s_tg[0];
    total_hess_e  = s_th[0];
    total_count_e = s_tc[0];
    const float era_weight  = *s_era_weight;
    const float parent_gain = *s_parent_gain;

    // prefix on first Tfull bins (we never split at the last bin)
    block_scan_prefix(grad_bins, hess_bins, count_bins, num_bins-1);

    // one (or more) thresholds per thread
    for (int t = threadIdx.x; t < Keval; t += blockDim.x) {
      const int thr = thr_sh[t];
      if (thr >= 0 && thr < (num_bins - 1) && era_weight > 0.f) {
        const int   lc = count_bins[thr];
        const float lg = grad_bins[thr];
        const float lh = hess_bins[thr];
        const int   rc = total_count_e - lc;
        if (lc > 0 && rc > 0) {
          const float rg = total_grad_e - lg;
          const float rh = total_hess_e - lh;
          const float dL = lh + lambda_l2, dR = rh + lambda_l2;
          const float gain = 0.5f * ((lg*lg)/dL + (rg*rg)/dR) - parent_gain;

          const float delta = gain - mean_arr[t];
          const float neww  = wsum_arr[t] + era_weight;
          const float mean2 = mean_arr[t] + (era_weight/neww) * delta;
          const float delta2= gain - mean2;

          M2_arr [t] += era_weight * delta * delta2;
          mean_arr[t]  = mean2;
          wsum_arr[t]  = neww;

          leftg_arr[t] += lg;
          lefth_arr[t] += lh;
          leftc_arr[t] += (int64_t)lc;

          if (direction_weight != 0.f) {
            const float Lv = -lg/dL, Rv = -rg/dR;
            dir_arr[t] += era_weight * ((Lv > Rv) ? 1.f : -1.f);
          }
        }
      }
    }
    __syncthreads();
  } // eras

  // pick best threshold for this (node,feature)
  if(threadIdx.x==0){
    float best_s=NEG_INF; int best_t=-1; float best_lg=0.f, best_lh=0.f; int64_t best_lc=0;
    const float   TG = total_grad_nodes[node_id];
    const float   TH = total_hess_nodes[node_id];
    const int64_t TC = total_count_nodes[node_id];
    (void)TG; (void)TH; // kept for parity/debug; not used here directly

    for(int i=0;i<Keval;++i){
      const int    th = thr_sh[i];
      if(th<0 || th>=num_bins-1) continue;
      const int64_t lc = leftc_arr[i];
      const int64_t rc = TC - lc;
      if(lc < (int64_t)min_samples_leaf || rc < (int64_t)min_samples_leaf) continue;

      const float wsum = wsum_arr[i];
      const float std  = (wsum>0.f) ? sqrtf(fmaxf(0.f, M2_arr[i]/wsum)) : 0.f;
      float score = mean_arr[i] - lambda_dro * std;
      if(direction_weight!=0.f && wsum>0.f) score += direction_weight * (dir_arr[i]/wsum);

      bool better = (score > best_s);
      if(!better && fabsf(score-best_s)<=1e-12f){
        if(lc>best_lc) better=true;
        else if(lc==best_lc && th>best_t) better=true;
      }
      if(better){ best_s=score; best_t=th; best_lg=leftg_arr[i]; best_lh=lefth_arr[i]; best_lc=lc; }
    }
    if(!(best_s==best_s) || isinf(best_s)){ best_t=-1; best_lg=0.f; best_lh=0.f; best_lc=0; }

    out_scores[out_index]=best_s;
    out_thresholds[out_index]=best_t;
    out_left_grad[out_index]=best_lg;
    out_left_hess[out_index]=best_lh;
    out_left_count[out_index]=best_lc;
  }
}

// ------------------------------- Host wrapper --------------------------------

py::dict find_best_splits_batched_cuda(
  py::object bins,               // torch.int8 [N,F] (row-major)
  py::object grad,               // torch.float32 [Rcat]
  py::object hess,               // torch.float32 [Rcat]
  py::object rows_index,         // torch.int32   [Rcat]
  py::object node_row_splits,    // torch.int32   [num_nodes+1]
  py::object node_era_splits,    // torch.int32   [num_nodes, num_eras+1]
  py::object era_weights,        // torch.float32 [num_nodes, num_eras]
  py::object total_grad,         // torch.float32 [num_nodes]
  py::object total_hess,         // torch.float32 [num_nodes]
  py::object total_count,        // torch.int64   [num_nodes]
  py::object feature_ids,        // torch.int32   [num_features]
  int max_bins,
  int k_cuts,
  const std::string& cut_selection,
  float lambda_l2,
  float lambda_dro,
  float direction_weight,
  int min_samples_leaf,
  int rows_total_compact         // == rows_index.shape[0]
){
  if(max_bins > MAX_BINS) throw std::invalid_argument("CUDA backend supports up to 128 bins per feature.");

  py::module torch = py::module::import("torch");

  // Shapes (feature-major [F,N])
  py::tuple bshape = py::tuple(bins.attr("shape"));   // [F,N]
  const int num_features_all = (int)py::int_(bshape[0]);
  const int rows_dataset     = (int)py::int_(bshape[1]);

  py::tuple nshape = py::tuple(node_row_splits.attr("shape"));
  const int num_nodes = (int)py::int_(nshape[0]) - 1;
  if(num_nodes<=0){
    return py::dict("scores"_a=py::none(), "thresholds"_a=py::none(),
                    "left_grad"_a=py::none(), "left_hess"_a=py::none(),
                    "left_count"_a=py::none(), "kernel_ms"_a=0.0);
  }

  py::tuple fshape = py::tuple(feature_ids.attr("shape"));
  const int num_features = (int)py::int_(fshape[0]);
  const int cut_mode = (cut_selection=="mass") ? 1 : 0;

  // Raw pointers — expect feature-major [F, N]
  const uintptr_t bins_ptr      = py::int_(bins.attr("data_ptr")());
  const uintptr_t grad_ptr      = py::int_(grad.attr("data_ptr")());
  const uintptr_t hess_ptr      = py::int_(hess.attr("data_ptr")());
  const uintptr_t rows_idx_ptr  = py::int_(rows_index.attr("data_ptr")());
  const uintptr_t nrow_ptr      = py::int_(node_row_splits.attr("data_ptr")());
  const uintptr_t nera_ptr      = py::int_(node_era_splits.attr("data_ptr")());
  const uintptr_t era_w_ptr     = py::int_(era_weights.attr("data_ptr")());
  const uintptr_t tgrad_ptr     = py::int_(total_grad.attr("data_ptr")());
  const uintptr_t thess_ptr     = py::int_(total_hess.attr("data_ptr")());
  const uintptr_t tcnt_ptr      = py::int_(total_count.attr("data_ptr")());
  const uintptr_t feats_ptr     = py::int_(feature_ids.attr("data_ptr")());

  // Outputs
  py::object device = bins.attr("device");
  auto scores_tensor = torch.attr("empty")(py::make_tuple(num_nodes, num_features),
                                           "device"_a=device, "dtype"_a=torch.attr("float32"));
  auto thresholds_tensor = torch.attr("empty")(py::make_tuple(num_nodes, num_features),
                                               "device"_a=device, "dtype"_a=torch.attr("int32"));
  auto left_grad_tensor = torch.attr("zeros")(py::make_tuple(num_nodes, num_features),
                                              "device"_a=device, "dtype"_a=torch.attr("float32"));
  auto left_hess_tensor = torch.attr("zeros")(py::make_tuple(num_nodes, num_features),
                                              "device"_a=device, "dtype"_a=torch.attr("float32"));
  auto left_count_tensor = torch.attr("zeros")(py::make_tuple(num_nodes, num_features),
                                               "device"_a=device, "dtype"_a=torch.attr("int64"));
  scores_tensor.attr("fill_")(-std::numeric_limits<float>::infinity());
  thresholds_tensor.attr("fill_")(-1);

  const uintptr_t out_s_ptr = py::int_(scores_tensor.attr("data_ptr")());
  const uintptr_t out_t_ptr = py::int_(thresholds_tensor.attr("data_ptr")());
  const uintptr_t out_g_ptr = py::int_(left_grad_tensor.attr("data_ptr")());
  const uintptr_t out_h_ptr = py::int_(left_hess_tensor.attr("data_ptr")());
  const uintptr_t out_c_ptr = py::int_(left_count_tensor.attr("data_ptr")());

  // Launch
  dim3 block(THREADS,1,1);
  dim3 grid((unsigned)num_features, (unsigned)num_nodes, 1);

  const int NB=max_bins;
  size_t shmem =
      (size_t)WARPS*NB*(sizeof(int32_t)+2*sizeof(float)) +   // per-warp hist
      (size_t)NB*(sizeof(int32_t)+2*sizeof(float)) +         // reduced hist
      (size_t)NB*sizeof(int32_t) +                           // count_total
      (size_t)NB*sizeof(int32_t) +                           // thr_sh
      (size_t)NB*(6*sizeof(float)+sizeof(int64_t)) +         // per-threshold accums
      (size_t)WARPS*(2*sizeof(float)+sizeof(int)) +          // per-warp totals
      (size_t)(2*sizeof(float)) +                            // broadcast slots
      (alignof(int64_t)-1);

  py::tuple era_shape = py::tuple(era_weights.attr("shape"));
  const int num_eras = (int)py::int_(era_shape[1]);

  cudaEvent_t start_evt, stop_evt;
  CUDA_CHECK(cudaEventCreate(&start_evt));
  CUDA_CHECK(cudaEventCreate(&stop_evt));
  CUDA_CHECK(cudaEventRecord(start_evt));

  cuda_find_best_splits_kernel<<<grid, block, shmem>>>(
    reinterpret_cast<const int8_t*>(bins_ptr),
    (int)rows_dataset,
    (int)num_features_all,
    reinterpret_cast<const float*>(grad_ptr),
    reinterpret_cast<const float*>(hess_ptr),
    reinterpret_cast<const int32_t*>(rows_idx_ptr),
    reinterpret_cast<const int32_t*>(nrow_ptr),
    reinterpret_cast<const int32_t*>(nera_ptr),
    reinterpret_cast<const float*>(era_w_ptr),
    reinterpret_cast<const float*>(tgrad_ptr),
    reinterpret_cast<const float*>(thess_ptr),
    reinterpret_cast<const int64_t*>(tcnt_ptr),
    reinterpret_cast<const int32_t*>(feats_ptr),
    (int)num_nodes,
    (int)num_features,
    (int)max_bins,
    (int)num_eras,
    (int)k_cuts,
    (int)((cut_selection=="mass")?1:0),
    (int)min_samples_leaf,
    (float)lambda_l2,
    (float)lambda_dro,
    (float)direction_weight,
    (int)rows_total_compact,
    reinterpret_cast<float*>(out_s_ptr),
    reinterpret_cast<int32_t*>(out_t_ptr),
    reinterpret_cast<float*>(out_g_ptr),
    reinterpret_cast<float*>(out_h_ptr),
    reinterpret_cast<int64_t*>(out_c_ptr)
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop_evt));
  CUDA_CHECK(cudaEventSynchronize(stop_evt));
  float kernel_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,start_evt,stop_evt));
  CUDA_CHECK(cudaEventDestroy(start_evt)); CUDA_CHECK(cudaEventDestroy(stop_evt));

  py::dict r;
  r["scores"]=scores_tensor; r["thresholds"]=thresholds_tensor;
  r["left_grad"]=left_grad_tensor; r["left_hess"]=left_hess_tensor;
  r["left_count"]=left_count_tensor; r["kernel_ms"]=kernel_ms;
  return r;
}

py::dict partition_frontier_cuda(
  py::object bins,               // torch.int8/uint8 [N,F] (row-major)
  py::object rows_index,         // torch.int32 [Rcat]
  py::object node_row_splits,    // torch.int32 [Nsel+1]
  py::object node_era_splits,    // torch.int32 [Nsel, E+1]
  py::object feat_per_node,      // torch.int32 [Nsel]
  py::object thr_per_node        // torch.int32 [Nsel]
){
  py::module torch = py::module::import("torch");

  // Shapes & meta (feature-major [F,N])
  py::tuple bshape = py::tuple(bins.attr("shape"));    // [F,N]
  const int N = (int)py::int_(bshape[1]);

  py::tuple nshape = py::tuple(node_row_splits.attr("shape")); // [Nsel+1]
  const int Nsel = (int)py::int_(nshape[0]) - 1;
  if (Nsel <= 0) {
    return py::dict("left_index"_a=torch.attr("empty")(py::make_tuple(0), "device"_a=bins.attr("device"),
                                                       "dtype"_a=torch.attr("int32")),
                    "right_index"_a=torch.attr("empty")(py::make_tuple(0), "device"_a=bins.attr("device"),
                                                        "dtype"_a=torch.attr("int32")),
                    "left_splits"_a=torch.attr("empty")(py::make_tuple(0,0), "device"_a=bins.attr("device"),
                                                        "dtype"_a=torch.attr("int32")),
                    "right_splits"_a=torch.attr("empty")(py::make_tuple(0,0), "device"_a=bins.attr("device"),
                                                         "dtype"_a=torch.attr("int32")));
  }

  py::tuple eshape = py::tuple(node_era_splits.attr("shape")); // [Nsel, E+1]
  const int E = (int)py::int_(eshape[1]) - 1;

  // Device & dtypes
  py::object device = bins.attr("device");
  auto opts_i32 = py::dict("device"_a=device, "dtype"_a=torch.attr("int32"));

  // Pointers — expect feature-major [F,N]
  const uintptr_t bins_ptr  = py::int_(bins.attr("data_ptr")());
  const uintptr_t rows_ptr  = py::int_(rows_index.attr("data_ptr")());
  const uintptr_t nrow_ptr  = py::int_(node_row_splits.attr("data_ptr")());
  const uintptr_t nera_ptr  = py::int_(node_era_splits.attr("data_ptr")());
  const uintptr_t feat_ptr  = py::int_(feat_per_node.attr("data_ptr")());
  const uintptr_t thr_ptr   = py::int_(thr_per_node.attr("data_ptr")());

  // Phase A: counts
  auto left_counts = torch.attr("zeros")(py::make_tuple(Nsel, E), **opts_i32);
  const uintptr_t lcnt_ptr = py::int_(left_counts.attr("data_ptr")());

  {
    dim3 grid((unsigned)Nsel, 1, 1);
    dim3 block(THREADS, 1, 1);
    frontier_count_kernel<THREADS><<<grid, block>>>(
      reinterpret_cast<const int8_t*>(bins_ptr),
      (int)N,
      reinterpret_cast<const int32_t*>(rows_ptr),
      reinterpret_cast<const int32_t*>(nrow_ptr),
      reinterpret_cast<const int32_t*>(nera_ptr),
      reinterpret_cast<const int32_t*>(feat_ptr),
      reinterpret_cast<const int32_t*>(thr_ptr),
      reinterpret_cast<int32_t*>(lcnt_ptr),
      (int)Nsel, (int)E
    );
    CUDA_CHECK(cudaGetLastError());
  }

  // Era lengths & splits (GPU, via torch ops)
  auto era_next = node_era_splits.attr("slice")(1, 1, E + 1);  // [Nsel,E]
  auto era_prev = node_era_splits.attr("slice")(1, 0, E);      // [Nsel,E]
  auto era_lens = era_next - era_prev;                         // [Nsel,E]

  auto right_counts = era_lens - left_counts;                  // [Nsel,E]
  auto zero_col     = torch.attr("zeros")(py::make_tuple(Nsel, 1), **opts_i32);

  // Per-node totals
  auto left_node_tot  = left_counts.attr("sum")(1).attr("contiguous")();   // [Nsel]
  auto right_node_tot = right_counts.attr("sum")(1).attr("contiguous")();  // [Nsel]

  // Local (per-node) splits [Nsel,E+1]
  auto left_cumsum  = left_counts.attr("cumsum")(1);
  auto right_cumsum = right_counts.attr("cumsum")(1);
  auto left_local   = torch.attr("cat")(py::make_tuple(zero_col, left_cumsum), 1).attr("contiguous")();
  auto right_local  = torch.attr("cat")(py::make_tuple(zero_col, right_cumsum), 1).attr("contiguous")();

  // Build per-node base offsets so that splits become ABSOLUTE into the flat outputs
  auto left_node_base = torch.attr("cat")(py::make_tuple(
      torch.attr("zeros")(py::make_tuple(1), **opts_i32),
      left_node_tot.attr("to")(torch.attr("int32")).attr("cumsum")(0).attr("slice")(0, 0, Nsel)
  ), 0).attr("contiguous")();  // [Nsel]
  auto right_node_base = torch.attr("cat")(py::make_tuple(
      torch.attr("zeros")(py::make_tuple(1), **opts_i32),
      right_node_tot.attr("to")(torch.attr("int32")).attr("cumsum")(0).attr("slice")(0, 0, Nsel)
  ), 0).attr("contiguous")();  // [Nsel]

  auto left_splits  = (left_local  + left_node_base.attr("unsqueeze")(1)).attr("contiguous")();   // [Nsel,E+1]
  auto right_splits = (right_local + right_node_base.attr("unsqueeze")(1)).attr("contiguous")();  // [Nsel,E+1]

  const long long total_left  = (long long)py::int_(left_node_tot.attr("sum")());
  const long long total_right = (long long)py::int_(right_node_tot.attr("sum")());

  auto out_left  = torch.attr("empty")(py::make_tuple(total_left),  **opts_i32);
  auto out_right = torch.attr("empty")(py::make_tuple(total_right), **opts_i32);

  // Phase B: scatter
  {
    const uintptr_t outL_ptr = py::int_(out_left.attr("data_ptr")());
    const uintptr_t outR_ptr = py::int_(out_right.attr("data_ptr")());
    const uintptr_t ls_ptr   = py::int_(left_splits.attr("data_ptr")());
    const uintptr_t rs_ptr   = py::int_(right_splits.attr("data_ptr")());

    dim3 grid((unsigned)Nsel, 1, 1);
    dim3 block(THREADS, 1, 1);
    size_t shmem = sizeof(int32_t) * (size_t)E * 2;   // curL[E] + curR[E]
    frontier_scatter_kernel<THREADS><<<grid, block, shmem>>>(
      reinterpret_cast<const int8_t*>(bins_ptr), (int)N,
      reinterpret_cast<const int32_t*>(rows_ptr),
      reinterpret_cast<const int32_t*>(nrow_ptr),
      reinterpret_cast<const int32_t*>(nera_ptr),
      reinterpret_cast<const int32_t*>(feat_ptr),
      reinterpret_cast<const int32_t*>(thr_ptr),
      reinterpret_cast<const int32_t*>(ls_ptr),
      reinterpret_cast<const int32_t*>(rs_ptr),
      reinterpret_cast<int32_t*>(outL_ptr),
      reinterpret_cast<int32_t*>(outR_ptr),
      (int)Nsel, (int)E
    );
    CUDA_CHECK(cudaGetLastError());
  }

  py::dict r;
  r["left_index"]  = out_left;
  r["right_index"] = out_right;
  r["left_splits"]  = left_splits;
  r["right_splits"] = right_splits;
  return r;
}

py::object predict_bins_cuda(
    py::object bins,        // torch.uint8 or int8 [N,F] on CUDA
    py::object feature,     // torch.int32 [num_nodes] on same device
    py::object threshold,   // torch.int32 [num_nodes]
    py::object left,        // torch.int32 [num_nodes]
    py::object right,       // torch.int32 [num_nodes]
    py::object value,       // torch.float32 [num_nodes]
    py::object is_leaf      // torch.bool [num_nodes]  (or uint8)
){
    py::module torch = py::module::import("torch");
    // bins is feature-major [F, N]
    py::tuple bshape = py::tuple(bins.attr("shape"));
    const int F = (int)py::int_(bshape[0]);
    const int N = (int)py::int_(bshape[1]);

    py::tuple nshape = py::tuple(feature.attr("shape"));
    const int num_nodes = (int)py::int_(nshape[0]);

    if (N == 0 || num_nodes == 0) {
        return torch.attr("empty")(py::make_tuple(0), "device"_a=bins.attr("device"),
                                   "dtype"_a=torch.attr("float32"));
    }

    // device
    py::object device = bins.attr("device");

    // Make sure dtypes are as expected (cheap, no copies if already correct)
    auto bins_u8 = bins;
    if (!py::bool_(bins.attr("dtype").attr("is_floating_point")()).cast<bool>()) {
        // cast to uint8 view if needed (int8 is fine; pointer is reinterpreted as uint8_t)
        // We’ll pass its data_ptr as uint8_t*
    }

    // output
    auto out = torch.attr("empty")(py::make_tuple(N), "device"_a=device,
                                   "dtype"_a=torch.attr("float32"));

    // raw ptrs — expect feature-major [F,N]
    const uintptr_t bins_ptr  = py::int_(bins.attr("data_ptr")());
    const uintptr_t feat_ptr  = py::int_(feature.attr("data_ptr")());
    const uintptr_t thr_ptr   = py::int_(threshold.attr("data_ptr")());
    const uintptr_t left_ptr  = py::int_(left.attr("data_ptr")());
    const uintptr_t right_ptr = py::int_(right.attr("data_ptr")());
    const uintptr_t val_ptr   = py::int_(value.attr("data_ptr")());
    // is_leaf: accept bool or uint8; we read bytes (0/1)
    const uintptr_t leaf_ptr  = py::int_(is_leaf.attr("to")(torch.attr("uint8")).attr("contiguous")().attr("data_ptr")());
    const uintptr_t out_ptr   = py::int_(out.attr("data_ptr")());

    // launch
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;
    blocks = std::min(blocks, 65535);  // 1D grid cap

    cuda_predict_bins_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint8_t*>(bins_ptr), N,
        reinterpret_cast<const int32_t*>(feat_ptr),
        reinterpret_cast<const int32_t*>(thr_ptr),
        reinterpret_cast<const int32_t*>(left_ptr),
        reinterpret_cast<const int32_t*>(right_ptr),
        reinterpret_cast<const uint8_t*>(leaf_ptr),
        reinterpret_cast<const float*>(val_ptr),
        (int)num_nodes,
        reinterpret_cast<float*>(out_ptr)
    );
    CUDA_CHECK(cudaGetLastError());

    return out;
}


// ================== Pack predictor: warp-parallel over trees ==================
// ================== Pack predictor: shared-mem reduction, no warp intrinsics ==================
__global__ void predict_pack_kernel(
  const uint8_t* __restrict__ bins, int F, int N,
  const int32_t* __restrict__ feature,     // [nodes_total_pack]
  const int32_t* __restrict__ threshold,   // [nodes_total_pack]
  const int32_t* __restrict__ left_abs,    // [nodes_total_pack]
  const int32_t* __restrict__ right_abs,   // [nodes_total_pack]
  const uint8_t* __restrict__ is_leaf,     // [nodes_total_pack] (0/1)
  const float*   __restrict__ value,       // [nodes_total_pack]
  const int32_t* __restrict__ offsets,     // [B+1]
  int B,                                   // trees in this pack
  float tree_weight,                       // common weight per tree
  float* __restrict__ out                  // [N] (+=)
){
  // One block cooperatively accumulates the sum over trees for a row,
  // using shared-memory reduction. Blocks iterate rows grid-stride.
  extern __shared__ float s_partials[]; // size: blockDim.x

  (void)F; // F not used here, but kept for parity

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    float partial = 0.0f;

    // Each thread processes a striped subset of trees
    for (int t = threadIdx.x; t < B; t += blockDim.x) {
      float leaf_val = 0.0f;

      // Traverse tree t (absolute indices within the packed arrays)
      int node      = offsets[t];
      const int end = offsets[t + 1];
      const int max_steps = end - node;  // defensive cap

      for (int it = 0; it < max_steps; ++it) {
        if (is_leaf[node]) { leaf_val = value[node]; break; }
        const int f   = feature[node];
        const int thr = threshold[node];
        const int v   = load_bin_colmajor(bins, N, row, f);  // bins is [F,N]
        const int nxt = (v <= thr) ? left_abs[node] : right_abs[node];
        // guard against corrupted children
        if (nxt < offsets[t] || nxt >= end) { leaf_val = 0.0f; break; }
        node = nxt;
      }

      partial += leaf_val;
    }

    // Block-wide reduction in shared memory
    s_partials[threadIdx.x] = partial;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        s_partials[threadIdx.x] += s_partials[threadIdx.x + offset];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      out[row] += tree_weight * s_partials[0];
    }
    __syncthreads(); // reuse s_partials safely next row
  }
}


// ================== Host wrapper (API unchanged) ==================
py::object predict_pack_cuda(
  py::object bins,         // torch.uint8/int8 [F,N] on CUDA (feature-major)
  py::object feature,      // torch.int32 [nodes_total_pack]
  py::object threshold,    // torch.int32 [nodes_total_pack]
  py::object left_abs,     // torch.int32 [nodes_total_pack]
  py::object right_abs,    // torch.int32 [nodes_total_pack]
  py::object value,        // torch.float32 [nodes_total_pack]
  py::object is_leaf,      // torch.bool/uint8 [nodes_total_pack]
  py::object offsets,      // torch.int32 [B+1]
  float tree_weight        // per-tree scale (e.g., lr / pack_size)
){
  py::module torch = py::module::import("torch");

  // bins is feature-major [F, N]
  py::tuple bshape = py::tuple(bins.attr("shape"));
  const int F = (int)py::int_(bshape[0]);
  const int N = (int)py::int_(bshape[1]);
  if (N == 0) {
    return torch.attr("empty")(py::make_tuple(0), "device"_a=bins.attr("device"),
                               "dtype"_a=torch.attr("float32"));
  }

  const int B = (int)py::int_(py::tuple(offsets.attr("shape"))[0]) - 1;

  auto out = torch.attr("zeros")(py::make_tuple(N),
                                 "device"_a=bins.attr("device"),
                                 "dtype"_a=torch.attr("float32"));

  // Expect feature-major [F,N] from caller
  const uintptr_t bins_ptr = py::int_(bins.attr("data_ptr")());
  const uintptr_t f_ptr    = py::int_(feature.attr("data_ptr")());
  const uintptr_t t_ptr    = py::int_(threshold.attr("data_ptr")());
  const uintptr_t l_ptr    = py::int_(left_abs.attr("data_ptr")());
  const uintptr_t r_ptr    = py::int_(right_abs.attr("data_ptr")());
  const uintptr_t v_ptr    = py::int_(value.attr("data_ptr")());
  const uintptr_t leaf_ptr = py::int_(is_leaf.attr("to")(torch.attr("uint8")).attr("contiguous")().attr("data_ptr")());
  const uintptr_t off_ptr  = py::int_(offsets.attr("data_ptr")());
  const uintptr_t out_ptr  = py::int_(out.attr("data_ptr")());

  // Launch config: cooperative reduction width = threads
  const int threads = 256;                               // tuneable: 128/256/512
  const int blocks  = std::min(65535, std::max(1, (N + threads - 1) / threads));
  const size_t shmem = sizeof(float) * (size_t)threads;  // one float per thread

  predict_pack_kernel<<<blocks, threads, shmem>>>(
      reinterpret_cast<const uint8_t*>(bins_ptr), F, N,
      reinterpret_cast<const int32_t*>(f_ptr),
      reinterpret_cast<const int32_t*>(t_ptr),
      reinterpret_cast<const int32_t*>(l_ptr),
      reinterpret_cast<const int32_t*>(r_ptr),
      reinterpret_cast<const uint8_t*>(leaf_ptr),
      reinterpret_cast<const float*>(v_ptr),
      reinterpret_cast<const int32_t*>(off_ptr),
      (int)B, (float)tree_weight,
      reinterpret_cast<float*>(out_ptr)
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());  // ensure completion before returning the tensor

  return out;
}



} // namespace



void register_cuda_frontier(py::module_& m){
  m.def("_cuda_available", [](){return true;}, "Native CUDA backend is available.");
  m.def("find_best_splits_batched_cuda", &find_best_splits_batched_cuda,
    py::arg("bins"), py::arg("grad"), py::arg("hess"), py::arg("rows_index"),
    py::arg("node_row_splits"), py::arg("node_era_splits"), py::arg("era_weights"),
    py::arg("total_grad"), py::arg("total_hess"), py::arg("total_count"), py::arg("feature_ids"),
    py::arg("max_bins"), py::arg("k_cuts"), py::arg("cut_selection"),
    py::arg("lambda_l2"), py::arg("lambda_dro"), py::arg("direction_weight"),
    py::arg("min_samples_leaf"), py::arg("rows_total_compact"),
    "Find best splits for a batch of nodes using the CUDA backend.");
  // ====== in register_cuda_frontier(...) add:
  m.def(
    "predict_bins_cuda",
    &predict_bins_cuda,
    py::arg("bins"),
    py::arg("feature"),
    py::arg("threshold"),
    py::arg("left"),
    py::arg("right"),
    py::arg("value"),
    py::arg("is_leaf"),
    "Fast CUDA route for a single tree over binned features."
  );
  m.def("predict_pack_cuda", &predict_pack_cuda,
    py::arg("bins"), py::arg("feature"), py::arg("threshold"),
    py::arg("left_abs"), py::arg("right_abs"), py::arg("value"),
    py::arg("is_leaf"), py::arg("offsets"), py::arg("tree_weight"),
    "Warp-parallel CUDA predictor over a pack of trees (sum and scale).");
  m.def("partition_frontier_cuda", &partition_frontier_cuda,
      py::arg("bins"), py::arg("rows_index"),
      py::arg("node_row_splits"), py::arg("node_era_splits"),
      py::arg("feat_per_node"),   py::arg("thr_per_node"),
      "Batched frontier partition (count + warp-aggregated scatter).");



}
