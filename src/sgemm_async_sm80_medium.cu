#include "sgemm_async_sm80_medium.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace sgblas::detail {
namespace {

static_assert(SGBLAS_SM80_MEDIUM_THREAD_ROWS > 0 &&
                  SGBLAS_SM80_MEDIUM_THREAD_ROWS <= 128,
              "SM80 medium thread rows must be in [1, 128]");
static_assert(SGBLAS_SM80_MEDIUM_STAGES == 2 ||
                  SGBLAS_SM80_MEDIUM_STAGES == 3,
              "SM80 medium stages must be 2 or 3");
static_assert(SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES == 0 ||
                  SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES == 128 ||
                  SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES == 256,
              "SM80 medium L2 prefetch must be 0, 128, or 256 bytes");
static_assert(SGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS >= 0);
static_assert(SGBLAS_SM80_MEDIUM_MAX_WIDE_CTAS >=
              SGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS);

constexpr int kTileRows = 128;
constexpr int kTileColumns = 64;
constexpr int kTileK = 32;
constexpr int kThreads = 256;
constexpr int kThreadRows = SGBLAS_SM80_MEDIUM_THREAD_ROWS;
constexpr int kThreadColumns = kThreads / kThreadRows;
constexpr int kOutputsRows = kTileRows / kThreadRows;
constexpr int kOutputsColumns = kTileColumns / kThreadColumns;
constexpr int kStages = SGBLAS_SM80_MEDIUM_STAGES;
constexpr int kATileElements = kTileRows * kTileK;
constexpr int kBTileElements = kTileColumns * kTileK;
constexpr int kSharedBytes =
    kStages * (kATileElements + kBTileElements) * sizeof(float);
constexpr unsigned int kMaximumGridY = 65535U;

static_assert(kThreads % kThreadRows == 0);
static_assert(kTileRows % kThreadRows == 0);
static_assert(kTileColumns % kThreadColumns == 0);
static_assert(kStages == 2 || kStages == 3);

enum class BetaMode {
  kZero,
  kOne,
  kGeneral,
};

__device__ __forceinline__ void copyAsync16(void *destination,
                                            const void *source) {
#if __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
#if SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES == 128
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :
               : "r"(shared_address), "l"(source));
#elif SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES == 256
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" :
               : "r"(shared_address), "l"(source));
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(source));
#endif
#else
  *reinterpret_cast<float4 *>(destination) =
      *reinterpret_cast<const float4 *>(source);
#endif
}

__device__ __forceinline__ void commitAsyncCopies() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void waitForAsyncCopies() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;");
#endif
}

__device__ __forceinline__ void waitForOnePendingAsyncGroup() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;");
#endif
}

__device__ __forceinline__ void loadTileAsync(
    float *a_tile, float *b_tile, int stage, int tile_row, int tile_column,
    int tile_k, const float *a, int lda, const float *b, int ldb) {
  constexpr int vector_width = 4;
  constexpr int a_vector_loads_per_thread =
      kATileElements / (vector_width * kThreads);
  constexpr int b_vector_loads_per_thread =
      kBTileElements / (vector_width * kThreads);
  const int thread_index = static_cast<int>(threadIdx.x);

#pragma unroll
  for (int vector_load = 0; vector_load < a_vector_loads_per_thread;
       ++vector_load) {
    const int vector_index = thread_index + vector_load * kThreads;
    constexpr int vectors_per_k = kTileRows / vector_width;
    const int local_k = vector_index / vectors_per_k;
    const int local_row = (vector_index % vectors_per_k) * vector_width;
    const float *source =
        a + static_cast<std::size_t>(tile_row + local_row) +
        static_cast<std::size_t>(tile_k + local_k) * lda;
    copyAsync16(a_tile + stage * kATileElements + local_k * kTileRows +
                    local_row,
                source);
  }

#pragma unroll
  for (int vector_load = 0; vector_load < b_vector_loads_per_thread;
       ++vector_load) {
    const int vector_index = thread_index + vector_load * kThreads;
    constexpr int vectors_per_column = kTileK / vector_width;
    const int local_column = vector_index / vectors_per_column;
    const int local_k =
        (vector_index % vectors_per_column) * vector_width;
    const float *source =
        b + static_cast<std::size_t>(tile_k + local_k) +
        static_cast<std::size_t>(tile_column + local_column) * ldb;
    copyAsync16(b_tile + stage * kBTileElements + local_column * kTileK +
                    local_k,
                source);
  }
  commitAsyncCopies();
}

template <BetaMode Mode, bool NMajorRaster>
__global__ __launch_bounds__(kThreads, 2) void sgemmAsyncNnMediumKernel(
    int m, int n, int k, float alpha, const float *a, int lda, const float *b,
    int ldb, float beta, float *c, int ldc) {
  extern __shared__ __align__(16) float shared[];
  float *a_tile = shared;
  float *b_tile = shared + kStages * kATileElements;

  const int thread_index = static_cast<int>(threadIdx.x);
  const int thread_row = thread_index % kThreadRows;
  const int thread_column = thread_index / kThreadRows;
  const int tile_row =
      static_cast<int>(NMajorRaster ? blockIdx.y : blockIdx.x) * kTileRows;
  const int tile_column =
      static_cast<int>(NMajorRaster ? blockIdx.x : blockIdx.y) * kTileColumns;

  float accumulators[kOutputsRows][kOutputsColumns] = {};

  int stage = 0;
  loadTileAsync(a_tile, b_tile, stage, tile_row, tile_column, 0, a, lda, b,
                ldb);
#if SGBLAS_SM80_MEDIUM_STAGES == 3
  if (k > kTileK) {
    loadTileAsync(a_tile, b_tile, 1, tile_row, tile_column, kTileK, a, lda, b,
                  ldb);
    waitForOnePendingAsyncGroup();
  } else {
    waitForAsyncCopies();
  }
#else
  waitForAsyncCopies();
#endif
  __syncthreads();

  for (int tile_k = 0; tile_k < k;) {
    const bool has_next_tile = k - tile_k > kTileK;
    const int next_tile_k = has_next_tile ? tile_k + kTileK : k;
#if SGBLAS_SM80_MEDIUM_STAGES == 3
    const bool has_following_tile = k - tile_k > 2 * kTileK;
    const int following_tile_k =
        has_following_tile ? tile_k + 2 * kTileK : k;
    if (has_following_tile) {
      loadTileAsync(a_tile, b_tile, (stage + 2) % kStages, tile_row,
                    tile_column, following_tile_k, a, lda, b, ldb);
    }
#else
    if (has_next_tile) {
      loadTileAsync(a_tile, b_tile, stage ^ 1, tile_row, tile_column,
                    next_tile_k, a, lda, b, ldb);
    }
#endif

#pragma unroll
    for (int reduction = 0; reduction < kTileK; ++reduction) {
      float a_values[kOutputsRows];
      float b_values[kOutputsColumns];
#pragma unroll
      for (int output_row = 0; output_row < kOutputsRows; ++output_row) {
        a_values[output_row] =
            a_tile[stage * kATileElements + reduction * kTileRows +
                   thread_row + output_row * kThreadRows];
      }
#pragma unroll
      for (int output_column = 0; output_column < kOutputsColumns;
           ++output_column) {
        b_values[output_column] =
            b_tile[stage * kBTileElements +
                   (thread_column + output_column * kThreadColumns) * kTileK +
                   reduction];
      }
#pragma unroll
      for (int output_column = 0; output_column < kOutputsColumns;
           ++output_column) {
#pragma unroll
        for (int output_row = 0; output_row < kOutputsRows; ++output_row) {
          accumulators[output_row][output_column] =
              fmaf(a_values[output_row], b_values[output_column],
                   accumulators[output_row][output_column]);
        }
      }
    }

    if (has_next_tile) {
#if SGBLAS_SM80_MEDIUM_STAGES == 3
      if (has_following_tile) {
        waitForOnePendingAsyncGroup();
      } else {
        waitForAsyncCopies();
      }
#else
      waitForAsyncCopies();
#endif
      __syncthreads();
#if SGBLAS_SM80_MEDIUM_STAGES == 3
      stage = (stage + 1) % kStages;
#else
      stage ^= 1;
#endif
      tile_k = next_tile_k;
    } else {
      break;
    }
  }

#pragma unroll
  for (int output_column = 0; output_column < kOutputsColumns;
       ++output_column) {
    const int global_column =
        tile_column + thread_column + output_column * kThreadColumns;
#pragma unroll
    for (int output_row = 0; output_row < kOutputsRows; ++output_row) {
      const int global_row =
          tile_row + thread_row + output_row * kThreadRows;
      const std::size_t c_index = static_cast<std::size_t>(global_row) +
                                  static_cast<std::size_t>(global_column) * ldc;
      const float product = alpha * accumulators[output_row][output_column];
      if constexpr (Mode == BetaMode::kZero) {
        c[c_index] = product;
      } else if constexpr (Mode == BetaMode::kOne) {
        c[c_index] += product;
      } else {
        c[c_index] = fmaf(beta, c[c_index], product);
      }
    }
  }
}

void prepareLaunchStatus() { (void)cudaGetLastError(); }

sgblasStatus_t launchStatus() {
  return cudaGetLastError() == cudaSuccess ? SGBLAS_STATUS_SUCCESS
                                           : SGBLAS_STATUS_EXECUTION_FAILED;
}

template <BetaMode Mode, bool NMajorRaster>
sgblasStatus_t launchKernel(dim3 grid, dim3 block, int m, int n, int k,
                            float alpha, const float *a, int lda,
                            const float *b, int ldb, float beta, float *c,
                            int ldc, cudaStream_t stream) {
  prepareLaunchStatus();
  if (cudaFuncSetAttribute(sgemmAsyncNnMediumKernel<Mode, NMajorRaster>,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           cudaSharedmemCarveoutMaxShared) != cudaSuccess) {
    return SGBLAS_STATUS_EXECUTION_FAILED;
  }
  if (cudaFuncSetAttribute(sgemmAsyncNnMediumKernel<Mode, NMajorRaster>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSharedBytes) != cudaSuccess) {
    return SGBLAS_STATUS_EXECUTION_FAILED;
  }
  sgemmAsyncNnMediumKernel<Mode, NMajorRaster>
      <<<grid, block, kSharedBytes, stream>>>(m, n, k, alpha, a, lda, b, ldb,
                                             beta, c, ldc);
  return launchStatus();
}

template <BetaMode Mode>
sgblasStatus_t launchMode(dim3 grid, bool n_major, dim3 block, int m, int n,
                          int k, float alpha, const float *a, int lda,
                          const float *b, int ldb, float beta, float *c, int ldc,
                          cudaStream_t stream) {
  if (n_major) {
    return launchKernel<Mode, true>(grid, block, m, n, k, alpha, a, lda, b, ldb,
                                    beta, c, ldc, stream);
  }
  return launchKernel<Mode, false>(grid, block, m, n, k, alpha, a, lda, b, ldb,
                                   beta, c, ldc, stream);
}

} // namespace

sgblasStatus_t launchAsyncNnSm80Medium(
    int m, int n, int k, float alpha, const float *a, int lda, const float *b,
    int ldb, float beta, float *c, int ldc, cudaStream_t stream) {
  const dim3 block(kThreads);
  const unsigned int row_tiles = static_cast<unsigned int>(m / kTileRows);
  const unsigned int column_tiles = static_cast<unsigned int>(n / kTileColumns);
  if (row_tiles > kMaximumGridY && column_tiles > kMaximumGridY) {
    return SGBLAS_STATUS_NOT_SUPPORTED;
  }
#if defined(SGBLAS_SM80_MEDIUM_N_MAJOR_RASTER)
  const bool n_major = row_tiles <= kMaximumGridY;
#else
  const bool n_major = column_tiles > kMaximumGridY;
#endif
  const dim3 grid = n_major ? dim3(column_tiles, row_tiles)
                            : dim3(row_tiles, column_tiles);
  if (beta == 0.0F) {
    return launchMode<BetaMode::kZero>(grid, n_major, block, m, n, k, alpha, a,
                                       lda, b, ldb, beta, c, ldc, stream);
  }
  if (beta == 1.0F) {
    return launchMode<BetaMode::kOne>(grid, n_major, block, m, n, k, alpha, a,
                                      lda, b, ldb, beta, c, ldc, stream);
  }
  return launchMode<BetaMode::kGeneral>(grid, n_major, block, m, n, k, alpha, a,
                                        lda, b, ldb, beta, c, ldc, stream);
}

} // namespace sgblas::detail
