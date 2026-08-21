#ifndef SGBLAS_SGBLAS_H_
#define SGBLAS_SGBLAS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgblasContext *sgblasHandle_t;

/* CUDA streams are opaque to the public C ABI so this header remains usable
 * in host-only builds that do not have the CUDA toolkit installed. */
typedef void *sgblasStream_t;

typedef enum sgblasStatus_t {
  SGBLAS_STATUS_SUCCESS = 0,
  SGBLAS_STATUS_NOT_INITIALIZED = 1,
  SGBLAS_STATUS_ALLOC_FAILED = 2,
  SGBLAS_STATUS_INVALID_VALUE = 3,
  SGBLAS_STATUS_ARCH_MISMATCH = 4,
  SGBLAS_STATUS_EXECUTION_FAILED = 5,
  SGBLAS_STATUS_INTERNAL_ERROR = 6,
  SGBLAS_STATUS_NOT_SUPPORTED = 7,
  SGBLAS_STATUS_FORCE_INT = 0x7fffffff
} sgblasStatus_t;

typedef enum sgblasOperation_t {
  SGBLAS_OP_N = 0,
  SGBLAS_OP_T = 1,
  SGBLAS_OP_FORCE_INT = 0x7fffffff
} sgblasOperation_t;

typedef enum sgblasMathMode_t {
  SGBLAS_MATH_FP32 = 0,
  SGBLAS_MATH_TF32 = 1,
  SGBLAS_MATH_FORCE_INT = 0x7fffffff
} sgblasMathMode_t;

sgblasStatus_t sgblasCreate(sgblasHandle_t *handle);
sgblasStatus_t sgblasDestroy(sgblasHandle_t handle);

sgblasStatus_t sgblasSetStream(sgblasHandle_t handle, sgblasStream_t stream);
sgblasStatus_t sgblasGetStream(sgblasHandle_t handle, sgblasStream_t *stream);

sgblasStatus_t sgblasSetMathMode(sgblasHandle_t handle, sgblasMathMode_t mode);
sgblasStatus_t sgblasGetMathMode(sgblasHandle_t handle, sgblasMathMode_t *mode);

/*
 * Column-major single-precision matrix multiplication:
 *
 *   C = (*alpha) * op(A) * op(B) + (*beta) * C
 *
 * alpha and beta are host pointers. Matrix pointers refer to device memory in
 * a CUDA build. The call enqueues work on the handle's stream and never
 * synchronizes it. When alpha is zero (or k is zero), A and B are not read.
 * When beta is zero, C is written without first being read.
 */
sgblasStatus_t sgblasSgemm(sgblasHandle_t handle, sgblasOperation_t transa,
                           sgblasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *a, int lda,
                           const float *b, int ldb, const float *beta, float *c,
                           int ldc);

const char *sgblasGetStatusString(sgblasStatus_t status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SGBLAS_SGBLAS_H_ */
