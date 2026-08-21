#include <sgblas/sgblas.h>

#include <string.h>

int main(void) {
  sgblasHandle_t handle = NULL;
  if (sgblasCreate(&handle) != SGBLAS_STATUS_SUCCESS || handle == NULL) {
    return 1;
  }

  if (strcmp(sgblasGetStatusString(SGBLAS_STATUS_SUCCESS), "success") != 0) {
    (void)sgblasDestroy(handle);
    return 2;
  }

  return sgblasDestroy(handle) == SGBLAS_STATUS_SUCCESS ? 0 : 3;
}
