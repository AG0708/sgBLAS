#ifndef SGBLAS_HANDLE_INTERNAL_HPP_
#define SGBLAS_HANDLE_INTERNAL_HPP_

#include <sgblas/sgblas.h>

struct sgblasContext {
  sgblasStream_t stream = nullptr;
  sgblasMathMode_t math_mode = SGBLAS_MATH_FP32;
};

#endif /* SGBLAS_HANDLE_INTERNAL_HPP_ */
