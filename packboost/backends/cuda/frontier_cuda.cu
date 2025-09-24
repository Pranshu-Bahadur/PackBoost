// frontier_cuda.cu  (row-major bins; no per-call transpose; 64-bit safe; fast hist/scan/score)
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
constexpr int THREADS   = 32;                  // 4 warps/block
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

// Inclusive block scan over first n bins (n<=MAX_BINS)
__device__ inline void block_scan_prefix(float* g,float* h,int* c,int n){
  for(int off=1; off<n; off<<=1){
    __syncthreads();
    int b=threadIdx.x;
    float gprev=(b>=off && b<n)?g[b-off]:0.f;
    float hprev=(b>=off && b<n)?h[b-off]:0.f;
    int   cprev=(b>=off && b<n)?c[b-off]:0;
    __syncthreads();
    if(b<n){ g[b]+=gprev; h[b]+=hprev; c[b]+=cprev; }
  }
  __syncthreads();
}

__global__ void cuda_find_best_splits_kernel(
  // bins is ROW-MAJOR: [rows_dataset, num_features]
  const int8_t* __restrict__ bins_rowmajor,
  int bins_stride,                                 // = num_features
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

  if(node_id>=num_nodes || foff>=num_features) return;

  const int feature_id = feature_ids[foff];
  const int out_index  = node_id * num_features + foff;

  if(feature_id<0 || feature_id>=bins_stride || num_bins<=1){
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
      const long long off = (long long)ridx * (long long)bins_stride + (long long)feature_id; // row-major
      const int bin = int((unsigned char)bins_rowmajor[off]);
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
    Keval = __shfl_sync(0xffffffff, Keval, 0);
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
    Keval = __shfl_sync(0xffffffff, Keval, 0);
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
      const long long off = (long long)ridx * (long long)bins_stride + (long long)feature_id;
      const int bin = int((unsigned char)bins_rowmajor[off]);
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

    // totals for era
    float tg=0.f, th=0.f; int tc=0;
    for(int b=threadIdx.x; b<num_bins; b+=blockDim.x){ tg+=grad_bins[b]; th+=hess_bins[b]; tc+=count_bins[b]; }
    tg=warp_reduce_sum(tg); th=warp_reduce_sum(th); tc=warp_reduce_sum_int(tc);

    float w_e=0.f; if(lane==0) w_e = era_weights[node_id* num_eras + era];
    w_e = __shfl_sync(0xffffffff, w_e, 0);

    float parent_gain=0.f; if(lane==0) parent_gain = 0.5f*(tg*tg)/(th+lambda_l2);
    parent_gain = __shfl_sync(0xffffffff, parent_gain, 0);

    // left prefix on first Tfull bins
    block_scan_prefix(grad_bins, hess_bins, count_bins, num_bins-1);

    // one thread per threshold
    if(threadIdx.x < Keval){
      const int thr = thr_sh[threadIdx.x];
      if(thr>=0 && thr<num_bins-1 && w_e>0.f){
        const int   lc = count_bins[thr];
        const float lg = grad_bins[thr];
        const float lh = hess_bins[thr];
        const int   rc = tc - lc;
        if(lc>0 && rc>0){
          const float rg = tg - lg;
          const float rh = th - lh;
          const float dL = lh + lambda_l2;
          const float dR = rh + lambda_l2;
          const float gain = 0.5f*((lg*lg)/dL + (rg*rg)/dR) - parent_gain;

          // Welford
          const float delta = gain - mean_arr[threadIdx.x];
          const float neww  = wsum_arr[threadIdx.x] + w_e;
          const float mean2 = mean_arr[threadIdx.x] + (w_e/neww)*delta;
          const float delta2= gain - mean2;

          M2_arr   [threadIdx.x] += w_e * delta * delta2;
          mean_arr [threadIdx.x]  = mean2;
          wsum_arr [threadIdx.x]  = neww;

          leftg_arr[threadIdx.x] += lg;
          lefth_arr[threadIdx.x] += lh;
          leftc_arr[threadIdx.x] += (int64_t)lc;

          if(direction_weight!=0.f){
            const float Lv = -lg/dL, Rv = -rg/dR;
            dir_arr[threadIdx.x] += w_e * ((Lv>Rv)?1.f:-1.f);
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

  // Shapes
  py::tuple bshape = py::tuple(bins.attr("shape"));   // [N,F]
  const int rows_dataset = (int)py::int_(bshape[0]);
  const int bins_stride  = (int)py::int_(bshape[1]);

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

  // Raw pointers
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
      (alignof(int64_t)-1);

  cudaEvent_t start_evt, stop_evt;
  CUDA_CHECK(cudaEventCreate(&start_evt));
  CUDA_CHECK(cudaEventCreate(&stop_evt));
  CUDA_CHECK(cudaEventRecord(start_evt));

  const int num_eras =
      (int)py::int_(py::tuple(era_weights.attr("shape"))[1]);

  cuda_find_best_splits_kernel<<<grid, block, shmem>>>(
    reinterpret_cast<const int8_t*>(bins_ptr),
    (int)bins_stride,
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
    (int)cut_mode,
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
}
