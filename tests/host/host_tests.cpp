#include <sgblas/sgblas.h>

#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;

void expectStatus(std::string_view name, sgblasStatus_t actual,
                  sgblasStatus_t expected) {
  if (actual != expected) {
    std::cerr << name << ": expected " << sgblasGetStatusString(expected)
              << ", got " << sgblasGetStatusString(actual) << '\n';
    ++failures;
  }
}

void expectTrue(std::string_view name, bool condition) {
  if (!condition) {
    std::cerr << name << ": expectation failed\n";
    ++failures;
  }
}

} // namespace

int main() {
  expectStatus("create rejects null output", sgblasCreate(nullptr),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("destroy rejects null handle", sgblasDestroy(nullptr),
               SGBLAS_STATUS_NOT_INITIALIZED);

  sgblasHandle_t handle = nullptr;
  expectStatus("create", sgblasCreate(&handle), SGBLAS_STATUS_SUCCESS);
  expectTrue("create initializes handle", handle != nullptr);

  sgblasStream_t stream = reinterpret_cast<void *>(std::uintptr_t{0x1234});
  expectStatus("set stream", sgblasSetStream(handle, stream),
               SGBLAS_STATUS_SUCCESS);
  sgblasStream_t observed_stream = nullptr;
  expectStatus("get stream", sgblasGetStream(handle, &observed_stream),
               SGBLAS_STATUS_SUCCESS);
  expectTrue("stream round trip", observed_stream == stream);
  expectStatus("get stream rejects null output",
               sgblasGetStream(handle, nullptr), SGBLAS_STATUS_INVALID_VALUE);

  sgblasMathMode_t mode = SGBLAS_MATH_TF32;
  expectStatus("get default math mode", sgblasGetMathMode(handle, &mode),
               SGBLAS_STATUS_SUCCESS);
  expectTrue("default math mode is FP32", mode == SGBLAS_MATH_FP32);
  expectStatus("set TF32", sgblasSetMathMode(handle, SGBLAS_MATH_TF32),
               SGBLAS_STATUS_SUCCESS);
  expectStatus("get TF32", sgblasGetMathMode(handle, &mode),
               SGBLAS_STATUS_SUCCESS);
  expectTrue("TF32 round trip", mode == SGBLAS_MATH_TF32);
  expectStatus("reject invalid math mode",
               sgblasSetMathMode(handle, static_cast<sgblasMathMode_t>(99)),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("restore FP32", sgblasSetMathMode(handle, SGBLAS_MATH_FP32),
               SGBLAS_STATUS_SUCCESS);

  constexpr float zero = 0.0F;
  constexpr float one = 1.0F;
  constexpr float two = 2.0F;
  const auto fake_const =
      reinterpret_cast<const float *>(std::uintptr_t{0x1000});
  auto fake_mutable = reinterpret_cast<float *>(std::uintptr_t{0x2000});

  expectStatus("sgemm rejects null handle",
               sgblasSgemm(nullptr, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, &one,
                           fake_const, 2, fake_const, 4, &zero, fake_mutable,
                           2),
               SGBLAS_STATUS_NOT_INITIALIZED);
  expectStatus("sgemm rejects invalid operation",
               sgblasSgemm(handle, static_cast<sgblasOperation_t>(9),
                           SGBLAS_OP_N, 2, 3, 4, &one, fake_const, 2,
                           fake_const, 4, &zero, fake_mutable, 2),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("sgemm rejects negative dimension",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, -1, 3, 4, &one,
                           fake_const, 2, fake_const, 4, &zero, fake_mutable,
                           2),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("NN validates lda",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, &one,
                           fake_const, 1, fake_const, 4, &zero, fake_mutable,
                           2),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("TN validates lda against k",
               sgblasSgemm(handle, SGBLAS_OP_T, SGBLAS_OP_N, 2, 3, 4, &one,
                           fake_const, 3, fake_const, 4, &zero, fake_mutable,
                           2),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("NT validates ldb against n",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_T, 2, 3, 4, &one,
                           fake_const, 2, fake_const, 2, &zero, fake_mutable,
                           2),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("sgemm validates ldc",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, &one,
                           fake_const, 2, fake_const, 4, &zero, fake_mutable,
                           1),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("sgemm rejects missing alpha",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, nullptr,
                           fake_const, 2, fake_const, 4, &zero, fake_mutable,
                           2),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("sgemm rejects missing product input",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, &one,
                           nullptr, 2, fake_const, 4, &zero, fake_mutable, 2),
               SGBLAS_STATUS_INVALID_VALUE);

  expectStatus("empty output quick return",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 0, 3, 4, nullptr,
                           nullptr, 1, nullptr, 4, nullptr, nullptr, 1),
               SGBLAS_STATUS_SUCCESS);
  expectStatus("alpha zero beta one reads no matrices",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, &zero,
                           nullptr, 2, nullptr, 4, &one, nullptr, 2),
               SGBLAS_STATUS_SUCCESS);
  expectStatus("k zero beta one reads no matrices",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 0, &two,
                           nullptr, 2, nullptr, 1, &one, nullptr, 2),
               SGBLAS_STATUS_SUCCESS);
  expectStatus("scale requires C output",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, &zero,
                           nullptr, 2, nullptr, 4, &zero, nullptr, 2),
               SGBLAS_STATUS_INVALID_VALUE);
  expectStatus("host scale reports unsupported",
               sgblasSgemm(handle, SGBLAS_OP_N, SGBLAS_OP_N, 2, 3, 4, &zero,
                           nullptr, 2, nullptr, 4, &two, fake_mutable, 2),
               SGBLAS_STATUS_NOT_SUPPORTED);
  expectStatus("valid host SGEMM reports unsupported",
               sgblasSgemm(handle, SGBLAS_OP_T, SGBLAS_OP_T, 2, 3, 4, &one,
                           fake_const, 4, fake_const, 3, &zero, fake_mutable,
                           2),
               SGBLAS_STATUS_NOT_SUPPORTED);

  expectStatus("destroy", sgblasDestroy(handle), SGBLAS_STATUS_SUCCESS);

  if (failures != 0) {
    std::cerr << failures << " host test(s) failed\n";
    return 1;
  }
  std::cout << "All sgBLAS host API and validation tests passed.\n";
  return 0;
}
