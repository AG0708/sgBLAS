#include "handle_internal.hpp"

#include <new>

extern "C" sgblasStatus_t sgblasCreate(sgblasHandle_t *handle) {
  if (handle == nullptr) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }

  *handle = new (std::nothrow) sgblasContext{};
  if (*handle == nullptr) {
    return SGBLAS_STATUS_ALLOC_FAILED;
  }
  return SGBLAS_STATUS_SUCCESS;
}

extern "C" sgblasStatus_t sgblasDestroy(sgblasHandle_t handle) {
  if (handle == nullptr) {
    return SGBLAS_STATUS_NOT_INITIALIZED;
  }
  delete handle;
  return SGBLAS_STATUS_SUCCESS;
}

extern "C" sgblasStatus_t sgblasSetStream(sgblasHandle_t handle,
                                          sgblasStream_t stream) {
  if (handle == nullptr) {
    return SGBLAS_STATUS_NOT_INITIALIZED;
  }
  handle->stream = stream;
  return SGBLAS_STATUS_SUCCESS;
}

extern "C" sgblasStatus_t sgblasGetStream(sgblasHandle_t handle,
                                          sgblasStream_t *stream) {
  if (handle == nullptr) {
    return SGBLAS_STATUS_NOT_INITIALIZED;
  }
  if (stream == nullptr) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }
  *stream = handle->stream;
  return SGBLAS_STATUS_SUCCESS;
}

extern "C" sgblasStatus_t sgblasSetMathMode(sgblasHandle_t handle,
                                            sgblasMathMode_t mode) {
  if (handle == nullptr) {
    return SGBLAS_STATUS_NOT_INITIALIZED;
  }
  if (mode != SGBLAS_MATH_FP32 && mode != SGBLAS_MATH_TF32) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }
  handle->math_mode = mode;
  return SGBLAS_STATUS_SUCCESS;
}

extern "C" sgblasStatus_t sgblasGetMathMode(sgblasHandle_t handle,
                                            sgblasMathMode_t *mode) {
  if (handle == nullptr) {
    return SGBLAS_STATUS_NOT_INITIALIZED;
  }
  if (mode == nullptr) {
    return SGBLAS_STATUS_INVALID_VALUE;
  }
  *mode = handle->math_mode;
  return SGBLAS_STATUS_SUCCESS;
}

extern "C" const char *sgblasGetStatusString(sgblasStatus_t status) {
  switch (status) {
  case SGBLAS_STATUS_SUCCESS:
    return "success";
  case SGBLAS_STATUS_NOT_INITIALIZED:
    return "not initialized";
  case SGBLAS_STATUS_ALLOC_FAILED:
    return "allocation failed";
  case SGBLAS_STATUS_INVALID_VALUE:
    return "invalid value";
  case SGBLAS_STATUS_ARCH_MISMATCH:
    return "architecture mismatch";
  case SGBLAS_STATUS_EXECUTION_FAILED:
    return "execution failed";
  case SGBLAS_STATUS_INTERNAL_ERROR:
    return "internal error";
  case SGBLAS_STATUS_NOT_SUPPORTED:
    return "not supported";
  case SGBLAS_STATUS_FORCE_INT:
    break;
  }
  return "unknown status";
}
