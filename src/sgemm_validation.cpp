#include "sgemm_validation.hpp"

#include <algorithm>

namespace sgblas::detail {
namespace {

bool validOperation(sgblasOperation_t operation) {
  return operation == SGBLAS_OP_N || operation == SGBLAS_OP_T;
}

} // namespace

sgblasStatus_t validateSgemm(sgblasOperation_t transa, sgblasOperation_t transb,
                             int m, int n, int k, const float *alpha,
                             const float *a, int lda, const float *b, int ldb,
                             const float *beta, float *c, int ldc,
                             GemmValidation *result) {
  if (result == nullptr) {
    return SGBLAS_STATUS_INTERNAL_ERROR;
  }
  *result = {};

  if (!validOperation(transa) || !validOperation(transb) || m < 0 || n < 0 ||
      k < 0) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }

  const int minimum_lda = std::max(1, transa == SGBLAS_OP_N ? m : k);
  const int minimum_ldb = std::max(1, transb == SGBLAS_OP_N ? k : n);
  const int minimum_ldc = std::max(1, m);
  if (lda < minimum_lda || ldb < minimum_ldb || ldc < minimum_ldc) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }

  // Empty output matrices are a true quick return and access no scalar or
  // matrix pointer, matching BLAS-style behavior.
  if (m == 0 || n == 0) {
    result->work = GemmWork::kNone;
    return SGBLAS_STATUS_SUCCESS;
  }

  if (alpha == nullptr || beta == nullptr) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }
  result->alpha = *alpha;
  result->beta = *beta;

  const bool has_product = k > 0 && result->alpha != 0.0F;
  if (!has_product) {
    if (result->beta == 1.0F) {
      result->work = GemmWork::kNone;
      return SGBLAS_STATUS_SUCCESS;
    }
    if (c == nullptr) {
      return SGBLAS_STATUS_INVALID_VALUE;
    }
    result->work = GemmWork::kScaleC;
    return SGBLAS_STATUS_SUCCESS;
  }

  if (a == nullptr || b == nullptr || c == nullptr) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }
  result->work = GemmWork::kProduct;
  return SGBLAS_STATUS_SUCCESS;
}

} // namespace sgblas::detail
