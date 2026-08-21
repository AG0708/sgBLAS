#include "handle_internal.hpp"
#include "sgemm_validation.hpp"

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

  // Host-only builds preserve API/validation behavior but cannot execute on
  // device pointers.
  return SGBLAS_STATUS_NOT_SUPPORTED;
}
