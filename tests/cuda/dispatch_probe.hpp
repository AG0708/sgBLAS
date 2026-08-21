#ifndef SGBLAS_TESTS_CUDA_DISPATCH_PROBE_HPP_
#define SGBLAS_TESTS_CUDA_DISPATCH_PROBE_HPP_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace sgblas::test {

enum class DispatchPath {
  kUnknown = 0,
  kScale,
  kGeneral,
  kShared,
  kRegister,
  kWide,
  kMedium,
  kSmall,
  kCount,
};

struct DispatchObservation {
  DispatchPath path = DispatchPath::kUnknown;
  std::string kernel_name;
  dim3 grid{};
  dim3 block{};
  std::size_t shared_bytes = 0;
};

DispatchObservation
captureDispatch(cudaStream_t stream, const std::function<void()> &launch);

const char *dispatchPathName(DispatchPath path);

std::string formatDispatchObservation(std::string_view case_name,
                                      const DispatchObservation &observation);

} // namespace sgblas::test

#endif /* SGBLAS_TESTS_CUDA_DISPATCH_PROBE_HPP_ */
