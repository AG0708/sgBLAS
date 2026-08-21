#ifndef SGBLAS_SGEMM_ASYNC_SM80_HPP_
#define SGBLAS_SGEMM_ASYNC_SM80_HPP_

#include <sgblas/sgblas.h>

#include <cuda_runtime_api.h>

namespace sgblas::detail {

sgblasStatus_t launchAsyncNnSm80(int m, int n, int k, float alpha,
                                 const float *a, int lda, const float *b,
                                 int ldb, float beta, float *c, int ldc,
                                 cudaStream_t stream);

} // namespace sgblas::detail

#endif /* SGBLAS_SGEMM_ASYNC_SM80_HPP_ */
