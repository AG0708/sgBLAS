#include <cuda_runtime_api.h>

#include <sgblas/sgblas.h>

#include "dispatch_probe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using sgblas::test::DispatchObservation;
using sgblas::test::DispatchPath;

struct TestCase {
  const char *name;
  sgblasOperation_t transa;
  sgblasOperation_t transb;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
  float alpha;
  float beta;
  DispatchPath expected_dispatch;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_sgblas(sgblasStatus_t status, std::string_view operation) {
  if (status != SGBLAS_STATUS_SUCCESS) {
    fail(std::string(operation) + " failed with sgBLAS status " +
         std::to_string(static_cast<int>(status)));
  }
}

class DispatchCoverage {
public:
  bool record(std::string_view case_name, DispatchPath expected,
              const DispatchObservation &observation) {
    std::cout << sgblas::test::formatDispatchObservation(case_name, observation)
              << '\n';
    if (observation.path == DispatchPath::kUnknown ||
        observation.path == DispatchPath::kCount) {
      std::cerr << case_name << ": unrecognized dispatch kernel \""
                << observation.kernel_name << "\"\n";
      return false;
    }

    ++hits_[static_cast<std::size_t>(observation.path)];
    if (expected != DispatchPath::kUnknown && expected != observation.path) {
      std::cerr << case_name << ": expected dispatch path "
                << sgblas::test::dispatchPathName(expected) << ", observed "
                << sgblas::test::dispatchPathName(observation.path) << '\n';
      return false;
    }
    return true;
  }

  bool require(DispatchPath path, bool required, std::string_view reason) const {
    const std::size_t hits = hits_[static_cast<std::size_t>(path)];
    std::cout << "DISPATCH_COVERAGE path="
              << sgblas::test::dispatchPathName(path) << " required="
              << (required ? "yes" : "no") << " hits=" << hits;
    if (!required) {
      std::cout << " SKIP reason=\"" << reason << "\"\n";
      return true;
    }
    const bool passed = hits != 0;
    std::cout << (passed ? " PASS\n" : " FAIL\n");
    return passed;
  }

private:
  std::array<unsigned int,
             static_cast<std::size_t>(DispatchPath::kCount)>
      hits_{};
};

class Stream {
public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  ~Stream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)),
               "cudaMalloc");
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  T *get() const { return data_; }

private:
  T *data_ = nullptr;
};

bool is_transposed(sgblasOperation_t operation) {
  return operation == SGBLAS_OP_T;
}

std::size_t stored_elements(int leading_dimension, int columns) {
  return static_cast<std::size_t>(leading_dimension) *
         static_cast<std::size_t>(columns);
}

double load_a(const std::vector<float> &matrix, const TestCase &test, int row,
              int reduction_index) {
  if (!is_transposed(test.transa)) {
    return static_cast<double>(matrix[row + reduction_index * test.lda]);
  }
  return static_cast<double>(matrix[reduction_index + row * test.lda]);
}

double load_b(const std::vector<float> &matrix, const TestCase &test,
              int reduction_index, int column) {
  if (!is_transposed(test.transb)) {
    return static_cast<double>(matrix[reduction_index + column * test.ldb]);
  }
  return static_cast<double>(matrix[column + reduction_index * test.ldb]);
}

void fill_random(std::vector<float> &values, std::mt19937 &generator) {
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  std::generate(values.begin(), values.end(),
                [&] { return distribution(generator); });
}

int minimum_lda(sgblasOperation_t operation, int m, int k) {
  return std::max(1, operation == SGBLAS_OP_N ? m : k);
}

int minimum_ldb(sgblasOperation_t operation, int n, int k) {
  return std::max(1, operation == SGBLAS_OP_N ? k : n);
}

#if defined(SGBLAS_TEST_HAS_SM80_ASYNC)
bool active_device_supports_async(std::size_t required_shared_bytes) {
  int device = -1;
  int compute_major = 0;
  int maximum_optin_shared_bytes = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice(dispatch coverage)");
  check_cuda(cudaDeviceGetAttribute(&compute_major,
                                    cudaDevAttrComputeCapabilityMajor, device),
             "cudaDeviceGetAttribute(compute capability)");
  check_cuda(cudaDeviceGetAttribute(&maximum_optin_shared_bytes,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    device),
             "cudaDeviceGetAttribute(opt-in shared memory)");
  return compute_major >= 8 &&
         static_cast<std::size_t>(maximum_optin_shared_bytes) >=
             required_shared_bytes;
}
#endif

void check_noop_case(std::string_view name, sgblasHandle_t handle,
                     cudaStream_t stream, sgblasOperation_t transa,
                     sgblasOperation_t transb, int m, int n, int k,
                     const float *alpha, const float *beta) {
  check_sgblas(sgblasSgemm(handle, transa, transb, m, n, k, alpha, nullptr,
                           minimum_lda(transa, m, k), nullptr,
                           minimum_ldb(transb, n, k), beta, nullptr,
                           std::max(1, m)),
               std::string("sgblasSgemm(") + std::string(name) + ')');
  // A successful enqueue is not enough: synchronize so an accidental kernel
  // launch that dereferences one of the null matrix pointers also fails here.
  check_cuda(cudaStreamSynchronize(stream),
             std::string("cudaStreamSynchronize(") + std::string(name) + ')');
  std::cout << std::left << std::setw(31) << name << std::right << " PASS\n";
}

bool run_scale_case(std::string_view name, int k, float alpha, float beta,
                    sgblasHandle_t handle, cudaStream_t stream,
                    std::mt19937 &generator, DispatchCoverage &coverage) {
  constexpr int m = 7;
  constexpr int n = 5;
  constexpr int lda = 11;
  const int ldb = std::max(1, k + 3);
  constexpr int ldc = 10;

  std::vector<float> initial_c(stored_elements(ldc, n));
  fill_random(initial_c, generator);
  if (beta == 0.0F) {
    // beta=0 promises not to read C. NaNs make a read-then-multiply-by-zero
    // implementation observably wrong instead of accidentally passing.
    for (int column = 0; column < n; ++column) {
      for (int row = 0; row < m; ++row) {
        initial_c[row + column * ldc] =
            std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  for (int column = 0; column < n; ++column) {
    for (int row = m; row < ldc; ++row) {
      initial_c[row + column * ldc] =
          211.0F + static_cast<float>(row) * 0.25F +
          static_cast<float>(column) * 0.03125F;
    }
  }

  DeviceBuffer<float> device_c(initial_c.size());
  check_cuda(cudaMemcpyAsync(device_c.get(), initial_c.data(),
                             initial_c.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(scale C)");
  const DispatchObservation dispatch = sgblas::test::captureDispatch(stream, [&] {
    check_sgblas(sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, m, n, k,
                             &alpha, nullptr, lda, nullptr, ldb, &beta,
                             device_c.get(), ldc),
                 std::string("sgblasSgemm(") + std::string(name) + ')');
  });
  const bool dispatch_passed =
      coverage.record(name, DispatchPath::kScale, dispatch);

  std::vector<float> actual(initial_c.size());
  check_cuda(cudaMemcpy(actual.data(), device_c.get(),
                        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy(scale C to host)");

  bool passed = true;
  constexpr double kTolerance = 2.0 * std::numeric_limits<float>::epsilon();
  for (int column = 0; column < n; ++column) {
    for (int row = 0; row < m; ++row) {
      const std::size_t index = static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * ldc;
      const double reference =
          beta == 0.0F ? 0.0
                       : static_cast<double>(beta) * initial_c[index];
      const double error =
          std::abs(static_cast<double>(actual[index]) - reference);
      const double tolerance = kTolerance * std::max(std::abs(reference), 1.0);
      if (!std::isfinite(actual[index]) || error > tolerance) {
        std::cerr << name << ": mismatch (row=" << row << ", col=" << column
                  << "): got " << actual[index] << ", reference " << reference
                  << '\n';
        passed = false;
      }
    }
    for (int row = m; row < ldc; ++row) {
      const std::size_t index = static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * ldc;
      if (actual[index] != initial_c[index]) {
        std::cerr << name << ": C padding overwritten (row=" << row
                  << ", col=" << column << ")\n";
        passed = false;
      }
    }
  }

  std::cout << std::left << std::setw(31) << name << std::right
            << (passed ? " PASS" : " FAIL") << '\n';
  return passed && dispatch_passed;
}

bool run_case(const TestCase &test, sgblasHandle_t handle, cudaStream_t stream,
              std::mt19937 &generator, DispatchCoverage &coverage) {
  const int a_columns = is_transposed(test.transa) ? test.m : test.k;
  const int b_columns = is_transposed(test.transb) ? test.k : test.n;

  std::vector<float> host_a(stored_elements(test.lda, a_columns));
  std::vector<float> host_b(stored_elements(test.ldb, b_columns));
  std::vector<float> initial_c(stored_elements(test.ldc, test.n));
  fill_random(host_a, generator);
  fill_random(host_b, generator);
  fill_random(initial_c, generator);

  if (test.beta == 0.0F) {
    // A beta-zero product must overwrite C without reading it.
    for (int column = 0; column < test.n; ++column) {
      for (int row = 0; row < test.m; ++row) {
        initial_c[row + column * test.ldc] =
            std::numeric_limits<float>::quiet_NaN();
      }
    }
  }

  // Give padding an unmistakable value so out-of-bounds row stores are caught.
  for (int column = 0; column < test.n; ++column) {
    for (int row = test.m; row < test.ldc; ++row) {
      initial_c[row + column * test.ldc] =
          123.0F + static_cast<float>(row) * 0.25F +
          static_cast<float>(column) * 0.03125F;
    }
  }

  DeviceBuffer<float> device_a(host_a.size());
  DeviceBuffer<float> device_b(host_b.size());
  DeviceBuffer<float> device_c(initial_c.size());

  check_cuda(cudaMemcpyAsync(device_a.get(), host_a.data(),
                             host_a.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(A)");
  check_cuda(cudaMemcpyAsync(device_b.get(), host_b.data(),
                             host_b.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(B)");
  check_cuda(cudaMemcpyAsync(device_c.get(), initial_c.data(),
                             initial_c.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(C)");

  const DispatchObservation dispatch = sgblas::test::captureDispatch(stream, [&] {
    check_sgblas(sgblasSgemm(handle, test.transa, test.transb, test.m, test.n,
                             test.k, &test.alpha, device_a.get(), test.lda,
                             device_b.get(), test.ldb, &test.beta,
                             device_c.get(), test.ldc),
                 std::string("sgblasSgemm(") + test.name + ')');
  });
  const bool dispatch_passed =
      coverage.record(test.name, test.expected_dispatch, dispatch);

  std::vector<float> actual(initial_c.size());
  check_cuda(cudaMemcpy(actual.data(), device_c.get(),
                        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy(C to host)");

  bool passed = true;
  int printed_mismatches = 0;
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;
  double maximum_error_ratio = 0.0;
  constexpr int kMaximumPrintedMismatches = 8;
  constexpr double kEpsilon = std::numeric_limits<float>::epsilon();

  for (int column = 0; column < test.n; ++column) {
    for (int row = 0; row < test.m; ++row) {
      double dot_product = 0.0;
      double absolute_product_sum = 0.0;
      for (int reduction_index = 0; reduction_index < test.k;
           ++reduction_index) {
        const double product = load_a(host_a, test, row, reduction_index) *
                               load_b(host_b, test, reduction_index, column);
        dot_product += product;
        absolute_product_sum += std::abs(product);
      }

      const std::size_t index = static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * test.ldc;
      const double product = static_cast<double>(test.alpha) * dot_product;
      const double reference =
          test.beta == 0.0F
              ? product
              : product + static_cast<double>(test.beta) * initial_c[index];
      const double observed = static_cast<double>(actual[index]);
      const double absolute_error = std::abs(observed - reference);
      const double relative_error =
          absolute_error / std::max(std::abs(reference), 1.0e-30);
      const double magnitude =
          std::abs(static_cast<double>(test.alpha)) * absolute_product_sum +
          (test.beta == 0.0F
               ? 0.0
               : std::abs(static_cast<double>(test.beta) * initial_c[index]));
      // A modest multiple of the standard FP32 dot-product forward-error
      // bound. This tolerates legal reassociation/FMA while still catching
      // indexing, transpose, and leading-dimension errors decisively.
      const double tolerance = 2.0e-6 + 8.0 * kEpsilon *
                                            static_cast<double>(test.k + 1) *
                                            std::max(magnitude, 1.0);
      const double error_ratio = absolute_error / tolerance;

      maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
      maximum_relative_error = std::max(maximum_relative_error, relative_error);
      maximum_error_ratio = std::max(maximum_error_ratio, error_ratio);

      if (!std::isfinite(observed) || absolute_error > tolerance) {
        passed = false;
        if (printed_mismatches++ < kMaximumPrintedMismatches) {
          std::cerr << "  mismatch (row=" << row << ", col=" << column
                    << "): got " << observed << ", reference " << reference
                    << ", abs error " << absolute_error << ", tolerance "
                    << tolerance << '\n';
        }
      }
    }
  }

  for (int column = 0; column < test.n; ++column) {
    for (int row = test.m; row < test.ldc; ++row) {
      const std::size_t index = static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * test.ldc;
      if (actual[index] != initial_c[index]) {
        passed = false;
        if (printed_mismatches++ < kMaximumPrintedMismatches) {
          std::cerr << "  C padding overwritten (row=" << row
                    << ", col=" << column << "): got " << actual[index]
                    << ", expected " << initial_c[index] << '\n';
        }
      }
    }
  }

  std::cout << std::left << std::setw(23) << test.name << std::right
            << " max_abs=" << std::scientific << std::setprecision(3)
            << maximum_absolute_error << " max_rel=" << maximum_relative_error
            << " bound_ratio=" << maximum_error_ratio << ' '
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed && dispatch_passed;
}

} // namespace

int main() {
  try {
#if defined(SGBLAS_TEST_FROZEN_SM80_HYBRID)
    constexpr std::size_t kExact512SmallSharedBytes =
        2U * (64U * 32U +
              static_cast<std::size_t>(SGBLAS_TEST_SM80_SMALL_TILE_COLUMNS) *
                  32U) *
        sizeof(float);
    const DispatchPath exact_512_expected_dispatch =
        active_device_supports_async(kExact512SmallSharedBytes)
            ? DispatchPath::kSmall
            : DispatchPath::kUnknown;
#else
    constexpr DispatchPath exact_512_expected_dispatch =
        DispatchPath::kUnknown;
#endif
    const std::vector<TestCase> tests = {
        {"NN odd beta=0", SGBLAS_OP_N, SGBLAS_OP_N, 37, 29, 43, 42, 50, 41,
         1.0F, 0.0F, DispatchPath::kShared},
        {"NT padded alpha/beta", SGBLAS_OP_N, SGBLAS_OP_T, 31, 27, 35, 36, 33,
         38, 0.75F, -0.25F, DispatchPath::kGeneral},
        {"TN padded alpha/beta", SGBLAS_OP_T, SGBLAS_OP_N, 25, 33, 39, 46, 44,
         31, -1.25F, 0.5F, DispatchPath::kGeneral},
        {"TT odd alpha/beta", SGBLAS_OP_T, SGBLAS_OP_T, 17, 23, 19, 24, 29, 22,
         0.375F, 1.25F, DispatchPath::kGeneral},
        {"NN alpha=0 beta!=0", SGBLAS_OP_N, SGBLAS_OP_N, 7, 5, 9, 11, 12, 10,
         0.0F, -0.75F, DispatchPath::kScale},
        {"NN register tile tails", SGBLAS_OP_N, SGBLAS_OP_N, 769, 771, 37, 774,
         42, 775, 0.875F, -0.125F, DispatchPath::kRegister},
        {"NN exact 512 small pocket", SGBLAS_OP_N, SGBLAS_OP_N, 512, 512, 512,
         512, 512, 512, 0.5F, -0.25F, exact_512_expected_dispatch},
        {"NN aligned full tiles", SGBLAS_OP_N, SGBLAS_OP_N, 768, 768, 64, 768,
         64, 768, -0.625F, 0.375F, DispatchPath::kUnknown},
        {"NN medium async tile", SGBLAS_OP_N, SGBLAS_OP_N, 1024, 2048, 64,
         1024, 64, 1024, 0.625F, -0.375F, DispatchPath::kUnknown},
        {"NN small async tile", SGBLAS_OP_N, SGBLAS_OP_N, 768, 768, 128, 768,
         128, 768, -0.375F, 0.625F, DispatchPath::kUnknown},
        // 1,048,561 rows require 65,536 16-row blocks in the transpose
        // fallback. This is intentionally skinny so the grid-limit regression
        // consumes only about 8 MiB across A, B, and C.
        {"NT skinny tall grid limit", SGBLAS_OP_N, SGBLAS_OP_T, 1048561, 1, 1,
         1048564, 3, 1048566, 1.0F, 0.0F, DispatchPath::kGeneral},
    };

    Stream stream;
    sgblasHandle_t handle = nullptr;
    check_sgblas(sgblasCreate(&handle), "sgblasCreate");

    bool all_passed = false;
    DispatchCoverage coverage;
    try {
      check_sgblas(
          sgblasSetStream(handle, reinterpret_cast<void *>(stream.get())),
          "sgblasSetStream");
      check_sgblas(sgblasSetMathMode(handle, SGBLAS_MATH_FP32),
                   "sgblasSetMathMode(FP32)");

      std::mt19937 generator(20260711U);
      all_passed = true;

      constexpr float zero = 0.0F;
      constexpr float one = 1.0F;
      constexpr float two = 2.0F;
      const sgblasOperation_t operations[] = {SGBLAS_OP_N, SGBLAS_OP_T};
      for (sgblasOperation_t transa : operations) {
        for (sgblasOperation_t transb : operations) {
          const std::string prefix =
              std::string(transa == SGBLAS_OP_N ? "N" : "T") +
              (transb == SGBLAS_OP_N ? "N" : "T");
          check_noop_case(prefix + " m=0 null no-op", handle, stream.get(),
                          transa, transb, 0, 7, 5, nullptr, nullptr);
          check_noop_case(prefix + " n=0 null no-op", handle, stream.get(),
                          transa, transb, 7, 0, 5, nullptr, nullptr);
        }
      }
      check_noop_case("alpha=0 beta=1 null no-op", handle, stream.get(),
                      SGBLAS_OP_N, SGBLAS_OP_N, 7, 5, 9, &zero, &one);
      check_noop_case("k=0 beta=1 null no-op", handle, stream.get(),
                      SGBLAS_OP_N, SGBLAS_OP_N, 7, 5, 0, &two, &one);

      all_passed = run_scale_case("alpha=0 beta=0 null A/B", 9, zero, zero,
                                  handle, stream.get(), generator, coverage) &&
                   all_passed;
      all_passed = run_scale_case("alpha=0 beta=general null A/B", 9, zero,
                                  -0.75F, handle, stream.get(), generator,
                                  coverage) &&
                   all_passed;
      all_passed = run_scale_case("k=0 beta=general null A/B", 0, two, -0.5F,
                                  handle, stream.get(), generator, coverage) &&
                   all_passed;

      for (const TestCase &test : tests) {
        all_passed = run_case(test, handle, stream.get(), generator, coverage) &&
                     all_passed;
      }

      all_passed = coverage.require(DispatchPath::kScale, true, {}) &&
                   all_passed;
      all_passed = coverage.require(DispatchPath::kGeneral, true, {}) &&
                   all_passed;
      all_passed = coverage.require(DispatchPath::kShared, true, {}) &&
                   all_passed;
      all_passed = coverage.require(DispatchPath::kRegister, true, {}) &&
                   all_passed;

#if defined(SGBLAS_TEST_HAS_SM80_ASYNC)
      constexpr std::size_t kWideSharedBytes = 4U * 128U * 32U * sizeof(float);
#if defined(SGBLAS_TEST_FROZEN_SM80_WIDE) ||                              \
    defined(SGBLAS_TEST_FROZEN_SM80_HYBRID)
      constexpr bool kFrozenWideDispatch = true;
#else
      constexpr bool kFrozenWideDispatch = false;
#endif
      const bool wide_device_supported =
          active_device_supports_async(kWideSharedBytes);
      const bool require_wide =
          kFrozenWideDispatch && wide_device_supported;
      const std::string_view wide_skip_reason =
          !kFrozenWideDispatch
              ? "non-frozen SM80 tuning can reroute the fixed wide probe"
              : "active device lacks SM80 or required opt-in shared memory";
      all_passed = coverage.require(
                       DispatchPath::kWide, require_wide, wide_skip_reason) &&
                   all_passed;
#else
      all_passed = coverage.require(DispatchPath::kWide, false,
                                    "SM80 wide kernel not compiled") &&
                   all_passed;
#endif

#if defined(SGBLAS_TEST_HAS_SM80_MEDIUM)
      constexpr std::size_t kMediumSharedBytes =
          static_cast<std::size_t>(SGBLAS_TEST_SM80_MEDIUM_STAGES) *
          (128U * 32U + 64U * 32U) * sizeof(float);
#if defined(SGBLAS_TEST_FROZEN_SM80_HYBRID)
      constexpr bool kFrozenMediumDispatch = true;
#else
      constexpr bool kFrozenMediumDispatch = false;
#endif
      const bool medium_device_supported =
          active_device_supports_async(kMediumSharedBytes);
      const bool require_medium =
          kFrozenMediumDispatch && medium_device_supported;
      const std::string_view medium_skip_reason =
          !kFrozenMediumDispatch
              ? "non-frozen SM80 tuning can reroute the fixed medium probe"
              : "active device lacks SM80 or required opt-in shared memory";
      all_passed = coverage.require(
                       DispatchPath::kMedium, require_medium,
                       medium_skip_reason) &&
                   all_passed;
#else
      all_passed = coverage.require(DispatchPath::kMedium, false,
                                    "SM80 medium kernel not compiled") &&
                   all_passed;
#endif

#if defined(SGBLAS_TEST_HAS_SM80_SMALL)
      constexpr std::size_t kSmallSharedBytes =
          2U * (64U * 32U +
                static_cast<std::size_t>(SGBLAS_TEST_SM80_SMALL_TILE_COLUMNS) *
                    32U) *
          sizeof(float);
#if defined(SGBLAS_TEST_FROZEN_SM80_HYBRID)
      constexpr bool kFrozenSmallDispatch = true;
#else
      constexpr bool kFrozenSmallDispatch = false;
#endif
      const bool small_device_supported =
          active_device_supports_async(kSmallSharedBytes);
      const bool require_small =
          kFrozenSmallDispatch && small_device_supported;
      const std::string_view small_skip_reason =
          !kFrozenSmallDispatch
              ? "non-frozen SM80 tuning can reroute the fixed small probe"
              : "active device lacks SM80 or required opt-in shared memory";
      all_passed = coverage.require(
                       DispatchPath::kSmall, require_small, small_skip_reason) &&
                   all_passed;
#else
      all_passed = coverage.require(DispatchPath::kSmall, false,
                                    "SM80 small kernel not compiled") &&
                   all_passed;
#endif
    } catch (...) {
      sgblasDestroy(handle);
      throw;
    }

    check_sgblas(sgblasDestroy(handle), "sgblasDestroy");
    if (!all_passed) {
      std::cerr << "One or more sgBLAS SGEMM correctness cases failed.\n";
      return EXIT_FAILURE;
    }

    std::cout << "All " << tests.size()
              << " matrix-product cases and 13 quick-return/scale cases "
                 "passed.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "correctness test error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
