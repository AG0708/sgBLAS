#include "sgemm_async_sm80.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace sgblas::detail {
namespace {

constexpr int kTileRows = 128;
constexpr int kTileColumns = 128;
constexpr int kTileK = 32;
constexpr int kThreads = 256;
constexpr int kThreadRows = 16;
constexpr int kThreadColumns = 16;
constexpr int kOutputsRows = kTileRows / kThreadRows;
constexpr int kOutputsColumns = kTileColumns / kThreadColumns;
constexpr int kTileElements = kTileRows * kTileK;
constexpr int kSharedBytes = 4 * kTileElements * sizeof(float);
constexpr unsigned int kMaximumGridY = 65535U;

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
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(source));
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

__device__ __forceinline__ void loadTileAsync(
    float *a_tile, float *b_tile, int stage, int tile_row, int tile_column,
    int tile_k, const float *a, int lda, const float *b, int ldb) {
  constexpr int vector_width = 4;
  constexpr int vector_loads_per_thread =
      kTileRows * kTileK / (vector_width * kThreads);
  const int thread_index = static_cast<int>(threadIdx.x);

#pragma unroll
  for (int vector_load = 0; vector_load < vector_loads_per_thread;
       ++vector_load) {
    const int vector_index = thread_index + vector_load * kThreads;

    constexpr int a_vectors_per_k = kTileRows / vector_width;
    const int a_local_k = vector_index / a_vectors_per_k;
    const int a_local_row = (vector_index % a_vectors_per_k) * vector_width;
    const float *a_source =
        a + static_cast<std::size_t>(tile_row + a_local_row) +
        static_cast<std::size_t>(tile_k + a_local_k) * lda;
    copyAsync16(a_tile + stage * kTileElements + a_local_k * kTileRows +
                    a_local_row,
                a_source);

    constexpr int b_vectors_per_column = kTileK / vector_width;
    const int b_local_column = vector_index / b_vectors_per_column;
    const int b_local_k =
        (vector_index % b_vectors_per_column) * vector_width;
    const float *b_source =
        b + static_cast<std::size_t>(tile_k + b_local_k) +
        static_cast<std::size_t>(tile_column + b_local_column) * ldb;
    copyAsync16(b_tile + stage * kTileElements +
                    b_local_column * kTileK + b_local_k,
                b_source);
  }
  commitAsyncCopies();
}

template <BetaMode Mode, bool NMajorRaster>
__global__ __launch_bounds__(kThreads, 1) void sgemmAsyncNnKernel(
    int m, int n, int k, float alpha, const float *a, int lda, const float *b,
    int ldb, float beta, float *c, int ldc) {
  extern __shared__ __align__(16) float shared[];
  float *a_tile = shared;
  float *b_tile = shared + 2 * kTileElements;

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
  waitForAsyncCopies();
  __syncthreads();

  for (int tile_k = 0; tile_k < k;) {
    const bool has_next_tile = k - tile_k > kTileK;
    const int next_tile_k = has_next_tile ? tile_k + kTileK : k;
    if (has_next_tile) {
      loadTileAsync(a_tile, b_tile, stage ^ 1, tile_row, tile_column,
                    next_tile_k, a, lda, b, ldb);
    }

#pragma unroll
    for (int reduction = 0; reduction < kTileK; ++reduction) {
      float a_values[kOutputsRows];
      float b_values[kOutputsColumns];
#pragma unroll
      for (int output_row = 0; output_row < kOutputsRows; ++output_row) {
        a_values[output_row] =
            a_tile[stage * kTileElements + reduction * kTileRows + thread_row +
                   output_row * kThreadRows];
      }
#pragma unroll
      for (int output_column = 0; output_column < kOutputsColumns;
           ++output_column) {
        b_values[output_column] =
            b_tile[stage * kTileElements +
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
      waitForAsyncCopies();
      __syncthreads();
      stage ^= 1;
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
  if (cudaFuncSetAttribute(sgemmAsyncNnKernel<Mode, NMajorRaster>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSharedBytes) != cudaSuccess) {
    return SGBLAS_STATUS_EXECUTION_FAILED;
  }
  sgemmAsyncNnKernel<Mode, NMajorRaster>
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

sgblasStatus_t launchAsyncNnSm80(int m, int n, int k, float alpha,
                                 const float *a, int lda, const float *b,
                                 int ldb, float beta, float *c, int ldc,
                                 cudaStream_t stream) {
  const dim3 block(kThreads);
  const unsigned int row_tiles = static_cast<unsigned int>(m / kTileRows);
  const unsigned int column_tiles = static_cast<unsigned int>(n / kTileColumns);
  if (row_tiles > kMaximumGridY && column_tiles > kMaximumGridY) {
    return SGBLAS_STATUS_NOT_SUPPORTED;
  }
  const bool n_major = column_tiles > kMaximumGridY;
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
