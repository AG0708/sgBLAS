#include "handle_internal.hpp"
#include "sgemm_validation.hpp"
#if defined(SGBLAS_EXPERIMENTAL_SM80_ASYNC)
#include "sgemm_async_sm80.hpp"
#endif
#if defined(SGBLAS_EXPERIMENTAL_SM80_MEDIUM)
#include "sgemm_async_sm80_medium.hpp"
#endif
#if defined(SGBLAS_EXPERIMENTAL_SM80_SMALL)
#include "sgemm_async_sm80_small.hpp"
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sgblas::detail {
namespace {

constexpr int kBlockRows = 16;
constexpr int kBlockColumns = 16;
constexpr int kSharedTileRows = 32;
constexpr int kSharedTileColumns = 32;
constexpr int kSharedTileK = 32;
constexpr int kSharedBlockColumns = 8;
constexpr int kSharedOutputsPerThread =
    kSharedTileColumns / kSharedBlockColumns;
constexpr int kRegisterTileRows = 128;
constexpr int kRegisterTileColumns = 128;
constexpr int kRegisterTileK = 32;
constexpr int kRegisterThreads = 256;
constexpr int kRegisterThreadRows = 16;
constexpr int kRegisterThreadColumns = 16;
constexpr int kRegisterOutputsRows = kRegisterTileRows / kRegisterThreadRows;
constexpr int kRegisterOutputsColumns =
    kRegisterTileColumns / kRegisterThreadColumns;
constexpr int kRegisterDispatchMinimum = 768;
constexpr unsigned int kMaximumGridY = 65535U;

#if defined(SGBLAS_EXPERIMENTAL_SM80_ASYNC)
constexpr int kAsyncWideSharedBytes =
    4 * kRegisterTileRows * kRegisterTileK * static_cast<int>(sizeof(float));
#endif
#if defined(SGBLAS_EXPERIMENTAL_SM80_MEDIUM)
constexpr int kAsyncMediumSharedBytes =
    SGBLAS_SM80_MEDIUM_STAGES *
    (kRegisterTileRows * kRegisterTileK + 64 * kRegisterTileK) *
    static_cast<int>(sizeof(float));
#endif
#if defined(SGBLAS_EXPERIMENTAL_SM80_SMALL)
constexpr int kAsyncSmallSharedBytes =
    2 * (64 * kRegisterTileK +
         SGBLAS_SM80_SMALL_TILE_COLUMNS * kRegisterTileK) *
    static_cast<int>(sizeof(float));
#endif

enum class BetaMode {
  kZero,
  kOne,
  kGeneral,
};

template <BetaMode Mode>
__global__ void scaleKernel(int m, int n, float beta, float *c, int ldc) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             static_cast<std::size_t>(threadIdx.x);
  if (column >= static_cast<std::size_t>(n)) {
    return;
  }

  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.y) * blockDim.y;
  for (std::size_t row =
           static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       row < static_cast<std::size_t>(m); row += row_stride) {
    const std::size_t index = row + column * static_cast<std::size_t>(ldc);
    if constexpr (Mode == BetaMode::kZero) {
      c[index] = 0.0F;
    } else {
      c[index] *= beta;
    }
  }
}

template <bool TransposeA, bool TransposeB, BetaMode Mode>
__global__ void sgemmNaiveKernel(int m, int n, int k, float alpha,
                                 const float *a, int lda, const float *b,
                                 int ldb, float beta, float *c, int ldc) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             static_cast<std::size_t>(threadIdx.x);
  if (column >= static_cast<std::size_t>(n)) {
    return;
  }

  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.y) * blockDim.y;
  for (std::size_t row =
           static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       row < static_cast<std::size_t>(m); row += row_stride) {
    float accumulator = 0.0F;
    for (int reduction = 0; reduction < k; ++reduction) {
      const std::size_t a_index =
          TransposeA ? static_cast<std::size_t>(reduction) +
                           row * static_cast<std::size_t>(lda)
                     : row + static_cast<std::size_t>(reduction) * lda;
      const std::size_t b_index =
          TransposeB ? column + static_cast<std::size_t>(reduction) * ldb
                     : static_cast<std::size_t>(reduction) +
                           column * static_cast<std::size_t>(ldb);
      accumulator = fmaf(a[a_index], b[b_index], accumulator);
    }

    const std::size_t c_index =
        row + column * static_cast<std::size_t>(ldc);
    const float product = alpha * accumulator;
    if constexpr (Mode == BetaMode::kZero) {
      c[c_index] = product;
    } else if constexpr (Mode == BetaMode::kOne) {
      c[c_index] += product;
    } else {
      c[c_index] = fmaf(beta, c[c_index], product);
    }
  }
}

// Column-major NN SGEMM. A 32x8 thread block computes a 32x32 C tile, with
// every thread accumulating four output columns for one row. Global loads are
// coalesced along the contiguous row dimension of A and reduction dimension of
// B. Padding keeps future transpose-style shared-memory accesses safe from
// 32-way bank conflicts without changing the fast broadcast access pattern.
template <BetaMode Mode, bool NMajorRaster>
__global__ __launch_bounds__(256) void sgemmSharedNnKernel(
    int m, int n, int k, float alpha, const float *a, int lda, const float *b,
    int ldb, float beta, float *c, int ldc) {
  __shared__ float a_tile[kSharedTileK][kSharedTileRows + 1];
  __shared__ float b_tile[kSharedTileColumns][kSharedTileK + 1];

  const int local_row = static_cast<int>(threadIdx.x);
  const int column_lane = static_cast<int>(threadIdx.y);
  const int thread_index = column_lane * kSharedTileRows + local_row;
  const int tile_row =
      static_cast<int>(NMajorRaster ? blockIdx.y : blockIdx.x) *
      kSharedTileRows;
  const int tile_column =
      static_cast<int>(NMajorRaster ? blockIdx.x : blockIdx.y) *
      kSharedTileColumns;
  const int global_row = tile_row + local_row;

  float accumulators[kSharedOutputsPerThread] = {};

  for (int tile_k = 0; tile_k < k;) {
#pragma unroll
    for (int linear = thread_index; linear < kSharedTileRows * kSharedTileK;
         linear += kSharedTileRows * kSharedBlockColumns) {
      const int a_row = linear % kSharedTileRows;
      const int a_k = linear / kSharedTileRows;
      const int global_a_row = tile_row + a_row;
      const int global_a_k = tile_k + a_k;
      a_tile[a_k][a_row] = global_a_row < m && global_a_k < k
                               ? a[static_cast<std::size_t>(global_a_row) +
                                   static_cast<std::size_t>(global_a_k) * lda]
                               : 0.0F;

      const int b_k = linear % kSharedTileK;
      const int b_column = linear / kSharedTileK;
      const int global_b_k = tile_k + b_k;
      const int global_b_column = tile_column + b_column;
      b_tile[b_column][b_k] =
          global_b_k < k && global_b_column < n
              ? b[static_cast<std::size_t>(global_b_k) +
                  static_cast<std::size_t>(global_b_column) * ldb]
              : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (int reduction = 0; reduction < kSharedTileK; ++reduction) {
      const float a_value = a_tile[reduction][local_row];
#pragma unroll
      for (int output = 0; output < kSharedOutputsPerThread; ++output) {
        const int local_column = column_lane + output * kSharedBlockColumns;
        accumulators[output] = fmaf(a_value, b_tile[local_column][reduction],
                                    accumulators[output]);
      }
    }
    __syncthreads();

    if (k - tile_k <= kSharedTileK) {
      break;
    }
    tile_k += kSharedTileK;
  }

  if (global_row >= m) {
    return;
  }
#pragma unroll
  for (int output = 0; output < kSharedOutputsPerThread; ++output) {
    const int global_column =
        tile_column + column_lane + output * kSharedBlockColumns;
    if (global_column >= n) {
      continue;
    }
    const std::size_t c_index = static_cast<std::size_t>(global_row) +
                                static_cast<std::size_t>(global_column) * ldc;
    const float product = alpha * accumulators[output];
    if constexpr (Mode == BetaMode::kZero) {
      c[c_index] = product;
    } else if constexpr (Mode == BetaMode::kOne) {
      c[c_index] += product;
    } else {
      c[c_index] = fmaf(beta, c[c_index], product);
    }
  }
}

// A 256-thread block computes a 128x128 C tile. Each thread holds an 8x8
// outer-product accumulator tile in registers. Rows and columns are interleaved
// by 16 across threads: for a fixed per-thread output index, half-warps access
// consecutive C rows while shared-memory A/B reads are conflict-free broadcasts
// or unit-stride transactions.
template <BetaMode Mode, bool NMajorRaster>
__global__ __launch_bounds__(kRegisterThreads, 2) void sgemmRegisterNnKernel(
    int m, int n, int k, float alpha, const float *a, int lda, const float *b,
    int ldb, float beta, float *c, int ldc) {
  __shared__ float a_tile[kRegisterTileK][kRegisterTileRows + 1];
  __shared__ float b_tile[kRegisterTileColumns][kRegisterTileK + 1];

  const int thread_index = static_cast<int>(threadIdx.x);
  const int thread_row = thread_index % kRegisterThreadRows;
  const int thread_column = thread_index / kRegisterThreadRows;
  const int tile_row =
      static_cast<int>(NMajorRaster ? blockIdx.y : blockIdx.x) *
      kRegisterTileRows;
  const int tile_column =
      static_cast<int>(NMajorRaster ? blockIdx.x : blockIdx.y) *
      kRegisterTileColumns;

  float accumulators[kRegisterOutputsRows][kRegisterOutputsColumns] = {};

  for (int tile_k = 0; tile_k < k;) {
    constexpr int vector_width = 4;
    constexpr int vector_loads_per_thread =
        kRegisterTileRows * kRegisterTileK / (vector_width * kRegisterThreads);
#pragma unroll
    for (int vector_load = 0; vector_load < vector_loads_per_thread;
         ++vector_load) {
        const int vector_index = thread_index + vector_load * kRegisterThreads;
        constexpr int a_vectors_per_k = kRegisterTileRows / vector_width;
        const int a_local_k = vector_index / a_vectors_per_k;
        const int a_local_row = (vector_index % a_vectors_per_k) * vector_width;
        const int a_global_k = tile_k + a_local_k;
        const int a_global_row = tile_row + a_local_row;
        float4 a_values = {};
        if (a_global_k < k && a_global_row <= m - vector_width) {
          const float *source = a + static_cast<std::size_t>(a_global_row) +
                                static_cast<std::size_t>(a_global_k) * lda;
          if ((reinterpret_cast<std::uintptr_t>(source) & 0xFU) == 0U) {
            a_values = *reinterpret_cast<const float4 *>(source);
          } else {
            a_values = make_float4(source[0], source[1], source[2], source[3]);
          }
        } else if (a_global_k < k) {
          const std::size_t column_offset =
              static_cast<std::size_t>(a_global_k) * lda;
          a_values.x = a_global_row + 0 < m
                           ? a[static_cast<std::size_t>(a_global_row + 0) +
                               column_offset]
                           : 0.0F;
          a_values.y = a_global_row + 1 < m
                           ? a[static_cast<std::size_t>(a_global_row + 1) +
                               column_offset]
                           : 0.0F;
          a_values.z = a_global_row + 2 < m
                           ? a[static_cast<std::size_t>(a_global_row + 2) +
                               column_offset]
                           : 0.0F;
          a_values.w = a_global_row + 3 < m
                           ? a[static_cast<std::size_t>(a_global_row + 3) +
                               column_offset]
                           : 0.0F;
        }
        a_tile[a_local_k][a_local_row + 0] = a_values.x;
        a_tile[a_local_k][a_local_row + 1] = a_values.y;
        a_tile[a_local_k][a_local_row + 2] = a_values.z;
        a_tile[a_local_k][a_local_row + 3] = a_values.w;

        constexpr int b_vectors_per_column = kRegisterTileK / vector_width;
        const int b_local_column = vector_index / b_vectors_per_column;
        const int b_local_k =
            (vector_index % b_vectors_per_column) * vector_width;
        const int b_global_column = tile_column + b_local_column;
        const int b_global_k = tile_k + b_local_k;
        float4 b_values = {};
        if (b_global_column < n && b_global_k <= k - vector_width) {
          const float *source = b + static_cast<std::size_t>(b_global_k) +
                                static_cast<std::size_t>(b_global_column) * ldb;
          if ((reinterpret_cast<std::uintptr_t>(source) & 0xFU) == 0U) {
            b_values = *reinterpret_cast<const float4 *>(source);
          } else {
            b_values = make_float4(source[0], source[1], source[2], source[3]);
          }
        } else if (b_global_column < n) {
          const std::size_t column_offset =
              static_cast<std::size_t>(b_global_column) * ldb;
          b_values.x =
              b_global_k + 0 < k
                  ? b[static_cast<std::size_t>(b_global_k + 0) + column_offset]
                  : 0.0F;
          b_values.y =
              b_global_k + 1 < k
                  ? b[static_cast<std::size_t>(b_global_k + 1) + column_offset]
                  : 0.0F;
          b_values.z =
              b_global_k + 2 < k
                  ? b[static_cast<std::size_t>(b_global_k + 2) + column_offset]
                  : 0.0F;
          b_values.w =
              b_global_k + 3 < k
                  ? b[static_cast<std::size_t>(b_global_k + 3) + column_offset]
                  : 0.0F;
        }
        b_tile[b_local_column][b_local_k + 0] = b_values.x;
        b_tile[b_local_column][b_local_k + 1] = b_values.y;
        b_tile[b_local_column][b_local_k + 2] = b_values.z;
        b_tile[b_local_column][b_local_k + 3] = b_values.w;
    }
    __syncthreads();

#pragma unroll
    for (int reduction = 0; reduction < kRegisterTileK; ++reduction) {
      float a_values[kRegisterOutputsRows];
      float b_values[kRegisterOutputsColumns];
#pragma unroll
      for (int output_row = 0; output_row < kRegisterOutputsRows;
           ++output_row) {
        a_values[output_row] =
            a_tile[reduction][thread_row + output_row * kRegisterThreadRows];
      }
#pragma unroll
      for (int output_column = 0; output_column < kRegisterOutputsColumns;
           ++output_column) {
        b_values[output_column] =
            b_tile[thread_column + output_column * kRegisterThreadColumns]
                  [reduction];
      }
#pragma unroll
      for (int output_column = 0; output_column < kRegisterOutputsColumns;
           ++output_column) {
#pragma unroll
        for (int output_row = 0; output_row < kRegisterOutputsRows;
             ++output_row) {
          accumulators[output_row][output_column] =
              fmaf(a_values[output_row], b_values[output_column],
                   accumulators[output_row][output_column]);
        }
      }
    }
    __syncthreads();

    if (k - tile_k <= kRegisterTileK) {
      break;
    }
    tile_k += kRegisterTileK;
  }

#pragma unroll
  for (int output_column = 0; output_column < kRegisterOutputsColumns;
       ++output_column) {
    const int global_column =
        tile_column + thread_column + output_column * kRegisterThreadColumns;
    if (global_column >= n) {
      continue;
    }
#pragma unroll
    for (int output_row = 0; output_row < kRegisterOutputsRows; ++output_row) {
      const int global_row =
          tile_row + thread_row + output_row * kRegisterThreadRows;
      if (global_row >= m) {
        continue;
      }
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

dim3 gridFor(int m, int n) {
  const unsigned int row_blocks =
      static_cast<unsigned int>((m - 1) / kBlockRows + 1);
  return dim3(static_cast<unsigned int>((n - 1) / kBlockColumns + 1),
              row_blocks < kMaximumGridY ? row_blocks : kMaximumGridY);
}

struct TiledGrid {
  dim3 dimensions{};
  bool n_major = false;
  bool valid = false;
};

unsigned int tileCount(int extent, int tile_extent) {
  return static_cast<unsigned int>(
      (static_cast<std::uint64_t>(extent) + tile_extent - 1U) /
      static_cast<unsigned int>(tile_extent));
}

TiledGrid tiledGridFor(int m, int n, int tile_rows, int tile_columns,
                       bool prefer_n_major = false) {
  const unsigned int row_tiles = tileCount(m, tile_rows);
  const unsigned int column_tiles = tileCount(n, tile_columns);
  if (prefer_n_major && row_tiles <= kMaximumGridY) {
    return {dim3(column_tiles, row_tiles), true, true};
  }
  if (column_tiles <= kMaximumGridY) {
    return {dim3(row_tiles, column_tiles), false, true};
  }
  if (row_tiles <= kMaximumGridY) {
    return {dim3(column_tiles, row_tiles), true, true};
  }
  return {};
}

bool fitsTiledGrid(int m, int n, int tile_rows, int tile_columns) {
  return tiledGridFor(m, n, tile_rows, tile_columns).valid;
}

void prepareLaunchStatus() { (void)cudaGetLastError(); }

sgblasStatus_t launchStatus() {
  return cudaGetLastError() == cudaSuccess ? SGBLAS_STATUS_SUCCESS
                                           : SGBLAS_STATUS_EXECUTION_FAILED;
}

#if defined(SGBLAS_EXPERIMENTAL_SM80_ASYNC)
struct AsyncDeviceCapabilities {
  int device = -1;
  int compute_major = 0;
  int maximum_optin_shared_bytes = 0;
};

bool activeDeviceSupportsAsync(int required_shared_bytes) {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }

  thread_local AsyncDeviceCapabilities capabilities;
  if (capabilities.device != device) {
    int compute_major = 0;
    int maximum_optin_shared_bytes = 0;
    if (cudaDeviceGetAttribute(&compute_major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&maximum_optin_shared_bytes,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device) != cudaSuccess) {
      return false;
    }
    capabilities = {device, compute_major, maximum_optin_shared_bytes};
  }
  return capabilities.compute_major >= 8 &&
         capabilities.maximum_optin_shared_bytes >= required_shared_bytes;
}
#endif

sgblasStatus_t launchScale(int m, int n, float beta, float *c, int ldc,
                           cudaStream_t stream) {
  const dim3 block(kBlockColumns, kBlockRows);
  const dim3 grid = gridFor(m, n);
  prepareLaunchStatus();
  if (beta == 0.0F) {
    scaleKernel<BetaMode::kZero>
        <<<grid, block, 0, stream>>>(m, n, beta, c, ldc);
  } else {
    scaleKernel<BetaMode::kGeneral>
        <<<grid, block, 0, stream>>>(m, n, beta, c, ldc);
  }
  return launchStatus();
}

template <bool TransposeA, bool TransposeB>
sgblasStatus_t launchProductByBeta(int m, int n, int k, float alpha,
                                   const float *a, int lda, const float *b,
                                   int ldb, float beta, float *c, int ldc,
                                   cudaStream_t stream) {
  const dim3 block(kBlockColumns, kBlockRows);
  const dim3 grid = gridFor(m, n);
  prepareLaunchStatus();
  if (beta == 0.0F) {
    sgemmNaiveKernel<TransposeA, TransposeB, BetaMode::kZero>
        <<<grid, block, 0, stream>>>(m, n, k, alpha, a, lda, b, ldb, beta, c,
                                     ldc);
  } else if (beta == 1.0F) {
    sgemmNaiveKernel<TransposeA, TransposeB, BetaMode::kOne>
        <<<grid, block, 0, stream>>>(m, n, k, alpha, a, lda, b, ldb, beta, c,
                                     ldc);
  } else {
    sgemmNaiveKernel<TransposeA, TransposeB, BetaMode::kGeneral>
        <<<grid, block, 0, stream>>>(m, n, k, alpha, a, lda, b, ldb, beta, c,
                                     ldc);
  }
  return launchStatus();
}

template <BetaMode Mode>
sgblasStatus_t launchSharedNnMode(const TiledGrid &grid, dim3 block, int m,
                                  int n, int k, float alpha, const float *a,
                                  int lda, const float *b, int ldb, float beta,
                                  float *c, int ldc, cudaStream_t stream) {
  prepareLaunchStatus();
  if (grid.n_major) {
    sgemmSharedNnKernel<Mode, true><<<grid.dimensions, block, 0, stream>>>(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    sgemmSharedNnKernel<Mode, false><<<grid.dimensions, block, 0, stream>>>(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  return launchStatus();
}

sgblasStatus_t launchSharedNnByBeta(int m, int n, int k, float alpha,
                                    const float *a, int lda, const float *b,
                                    int ldb, float beta, float *c, int ldc,
                                    cudaStream_t stream) {
  const dim3 block(kSharedTileRows, kSharedBlockColumns);
  const TiledGrid grid =
      tiledGridFor(m, n, kSharedTileRows, kSharedTileColumns);
  if (!grid.valid) {
    return launchProductByBeta<false, false>(m, n, k, alpha, a, lda, b, ldb,
                                             beta, c, ldc, stream);
  }
  if (beta == 0.0F) {
    return launchSharedNnMode<BetaMode::kZero>(
        grid, block, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream);
  }
  if (beta == 1.0F) {
    return launchSharedNnMode<BetaMode::kOne>(
        grid, block, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream);
  }
  return launchSharedNnMode<BetaMode::kGeneral>(
      grid, block, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream);
}

template <BetaMode Mode>
sgblasStatus_t launchRegisterNnMode(const TiledGrid &grid, dim3 block, int m,
                                    int n, int k, float alpha, const float *a,
                                    int lda, const float *b, int ldb,
                                    float beta, float *c, int ldc,
                                    cudaStream_t stream) {
  prepareLaunchStatus();
  if (grid.n_major) {
    sgemmRegisterNnKernel<Mode, true><<<grid.dimensions, block, 0, stream>>>(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    sgemmRegisterNnKernel<Mode, false><<<grid.dimensions, block, 0, stream>>>(
        m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  return launchStatus();
}

sgblasStatus_t launchRegisterNnByBeta(int m, int n, int k, float alpha,
                                      const float *a, int lda, const float *b,
                                      int ldb, float beta, float *c, int ldc,
                                      cudaStream_t stream) {
  const dim3 block(kRegisterThreads);
  const TiledGrid grid =
      tiledGridFor(m, n, kRegisterTileRows, kRegisterTileColumns);
  if (!grid.valid) {
    return launchProductByBeta<false, false>(m, n, k, alpha, a, lda, b, ldb,
                                             beta, c, ldc, stream);
  }
  if (beta == 0.0F) {
    return launchRegisterNnMode<BetaMode::kZero>(
        grid, block, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream);
  }
  if (beta == 1.0F) {
    return launchRegisterNnMode<BetaMode::kOne>(
        grid, block, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream);
  }
  return launchRegisterNnMode<BetaMode::kGeneral>(
      grid, block, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream);
}

sgblasStatus_t launchProduct(sgblasOperation_t transa, sgblasOperation_t transb,
                             int m, int n, int k, float alpha, const float *a,
                             int lda, const float *b, int ldb, float beta,
                             float *c, int ldc, cudaStream_t stream) {
  if (transa == SGBLAS_OP_N && transb == SGBLAS_OP_N) {
    if (m >= kRegisterDispatchMinimum && n >= kRegisterDispatchMinimum &&
        k >= kRegisterTileK) {
#if defined(SGBLAS_EXPERIMENTAL_SM80_ASYNC)
      const bool full_tiles = m % kRegisterTileRows == 0 &&
                              n % kRegisterTileColumns == 0 &&
                              k % kRegisterTileK == 0;
      const bool aligned = (reinterpret_cast<std::uintptr_t>(a) & 0xFU) == 0U &&
                           (reinterpret_cast<std::uintptr_t>(b) & 0xFU) == 0U &&
                           lda % 4 == 0 && ldb % 4 == 0;
      const std::size_t wide_ctas =
          static_cast<std::size_t>(m / kRegisterTileRows) *
          static_cast<std::size_t>(n / kRegisterTileColumns);
#if defined(SGBLAS_EXPERIMENTAL_SM80_SMALL)
      constexpr int kSmallTileRows = 64;
      constexpr int kSmallTileColumns = SGBLAS_SM80_SMALL_TILE_COLUMNS;
      const bool small_full_tiles = m % kSmallTileRows == 0 &&
                                    n % kSmallTileColumns == 0 &&
                                    k % kRegisterTileK == 0;
      const bool small_first_pocket =
          wide_ctas <= static_cast<std::size_t>(
                           SGBLAS_SM80_SMALL_MAX_WIDE_CTAS);
      const bool small_second_pocket =
          m <= SGBLAS_SM80_SMALL_SECOND_MAX_M &&
          n <= SGBLAS_SM80_SMALL_SECOND_MAX_N &&
          wide_ctas >= static_cast<std::size_t>(
                           SGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS) &&
          wide_ctas <= static_cast<std::size_t>(
                           SGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS);
      if (small_full_tiles && aligned && k >= SGBLAS_SM80_SMALL_MIN_K &&
          fitsTiledGrid(m, n, kSmallTileRows, kSmallTileColumns) &&
          activeDeviceSupportsAsync(kAsyncSmallSharedBytes) &&
          (small_first_pocket || small_second_pocket)) {
        return launchAsyncNnSm80Small(m, n, k, alpha, a, lda, b, ldb, beta, c,
                                     ldc, stream);
      }
#endif
#if defined(SGBLAS_EXPERIMENTAL_SM80_MEDIUM)
      constexpr int kMediumTileColumns = 64;
      const bool medium_full_tiles = m % kRegisterTileRows == 0 &&
                                     n % kMediumTileColumns == 0 &&
                                     k % kRegisterTileK == 0;
      if (medium_full_tiles && aligned &&
          k >= 2 * kRegisterTileK &&
          fitsTiledGrid(m, n, kRegisterTileRows, kMediumTileColumns) &&
          activeDeviceSupportsAsync(kAsyncMediumSharedBytes) &&
          wide_ctas >= static_cast<std::size_t>(
                           SGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS) &&
          wide_ctas <= static_cast<std::size_t>(
                           SGBLAS_SM80_MEDIUM_MAX_WIDE_CTAS)) {
        return launchAsyncNnSm80Medium(m, n, k, alpha, a, lda, b, ldb, beta, c,
                                      ldc, stream);
      }
#endif
      if (full_tiles && aligned &&
          fitsTiledGrid(m, n, kRegisterTileRows, kRegisterTileColumns) &&
          activeDeviceSupportsAsync(kAsyncWideSharedBytes)) {
        return launchAsyncNnSm80(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                 stream);
      }
#endif
      return launchRegisterNnByBeta(m, n, k, alpha, a, lda, b, ldb, beta, c,
                                    ldc, stream);
    }
    return launchSharedNnByBeta(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                stream);
  }
  if (transa == SGBLAS_OP_N && transb == SGBLAS_OP_T) {
    return launchProductByBeta<false, true>(m, n, k, alpha, a, lda, b, ldb,
                                            beta, c, ldc, stream);
  }
  if (transa == SGBLAS_OP_T && transb == SGBLAS_OP_N) {
    return launchProductByBeta<true, false>(m, n, k, alpha, a, lda, b, ldb,
                                            beta, c, ldc, stream);
  }
  return launchProductByBeta<true, true>(m, n, k, alpha, a, lda, b, ldb, beta,
                                         c, ldc, stream);
}

} // namespace
} // namespace sgblas::detail

extern "C" sgblasStatus_t sgblasSgemm(sgblasHandle_t handle,
                                      sgblasOperation_t transa,
                                      sgblasOperation_t transb, int m, int n,
                                      int k, const float *alpha, const float *a,
                                      int lda, const float *b, int ldb,
                                      const float *beta, float *c, int ldc) {
  if (handle == nullptr) {
    return SGBLAS_STATUS_NOT_INITIALIZED;
  }

  sgblas::detail::GemmValidation validation;
  const sgblasStatus_t status =
      sgblas::detail::validateSgemm(transa, transb, m, n, k, alpha, a, lda, b,
                                    ldb, beta, c, ldc, &validation);
  if (status != SGBLAS_STATUS_SUCCESS) {
    return status;
  }
  if (validation.work == sgblas::detail::GemmWork::kNone) {
    return SGBLAS_STATUS_SUCCESS;
  }
  if (handle->math_mode != SGBLAS_MATH_FP32) {
    return SGBLAS_STATUS_NOT_SUPPORTED;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(handle->stream);
  if (validation.work == sgblas::detail::GemmWork::kScaleC) {
    return sgblas::detail::launchScale(m, n, validation.beta, c, ldc, stream);
  }
  return sgblas::detail::launchProduct(transa, transb, m, n, k,
                                       validation.alpha, a, lda, b, ldb,
                                       validation.beta, c, ldc, stream);
}
