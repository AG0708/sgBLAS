#ifndef SGBLAS_SGEMM_VALIDATION_HPP_
#define SGBLAS_SGEMM_VALIDATION_HPP_

#include <sgblas/sgblas.h>

namespace sgblas::detail {

enum class GemmWork {
  kNone,
  kScaleC,
  kProduct,
};

struct GemmValidation {
  GemmWork work = GemmWork::kNone;
  float alpha = 0.0F;
  float beta = 0.0F;
};

sgblasStatus_t validateSgemm(sgblasOperation_t transa, sgblasOperation_t transb,
                             int m, int n, int k, const float *alpha,
                             const float *a, int lda, const float *b, int ldb,
                             const float *beta, float *c, int ldc,
                             GemmValidation *result);

} // namespace sgblas::detail

#endif /* SGBLAS_SGEMM_VALIDATION_HPP_ */
