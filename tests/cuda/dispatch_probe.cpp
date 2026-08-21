#include "dispatch_probe.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace sgblas::test {
namespace {

[[noreturn]] void failCuda(const char *operation, cudaError_t status) {
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(status));
}

void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    failCuda(operation, status);
  }
}

class GraphResources {
public:
  GraphResources() = default;
  GraphResources(const GraphResources &) = delete;
  GraphResources &operator=(const GraphResources &) = delete;

  ~GraphResources() {
    if (executable != nullptr) {
      cudaGraphExecDestroy(executable);
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
};

DispatchPath classifyKernel(std::string_view kernel_name) {
  if (kernel_name.find("scaleKernel") != std::string_view::npos) {
    return DispatchPath::kScale;
  }
  if (kernel_name.find("sgemmNaiveKernel") != std::string_view::npos) {
    return DispatchPath::kGeneral;
  }
  if (kernel_name.find("sgemmSharedNnKernel") != std::string_view::npos) {
    return DispatchPath::kShared;
  }
  if (kernel_name.find("sgemmRegisterNnKernel") != std::string_view::npos) {
    return DispatchPath::kRegister;
  }
  if (kernel_name.find("sgemmAsyncNnMediumKernel") != std::string_view::npos) {
    return DispatchPath::kMedium;
  }
  if (kernel_name.find("sgemmAsyncNnSmallKernel") != std::string_view::npos) {
    return DispatchPath::kSmall;
  }
  if (kernel_name.find("sgemmAsyncNnKernel") != std::string_view::npos) {
    return DispatchPath::kWide;
  }
  return DispatchPath::kUnknown;
}

} // namespace

DispatchObservation
captureDispatch(cudaStream_t stream, const std::function<void()> &launch) {
  // Keep pre-capture copies outside the graph so each observation contains
  // exactly the kernel enqueued by one sgblasSgemm call.
  checkCuda(cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(before dispatch capture)");
  checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
            "cudaStreamBeginCapture");

  GraphResources resources;
  try {
    launch();
  } catch (...) {
    cudaGraph_t abandoned_graph = nullptr;
    (void)cudaStreamEndCapture(stream, &abandoned_graph);
    if (abandoned_graph != nullptr) {
      cudaGraphDestroy(abandoned_graph);
    }
    throw;
  }

  checkCuda(cudaStreamEndCapture(stream, &resources.graph),
            "cudaStreamEndCapture");

  std::size_t node_count = 0;
  checkCuda(cudaGraphGetNodes(resources.graph, nullptr, &node_count),
            "cudaGraphGetNodes(count)");
  std::vector<cudaGraphNode_t> nodes(node_count);
  checkCuda(cudaGraphGetNodes(resources.graph, nodes.data(), &node_count),
            "cudaGraphGetNodes(nodes)");

  cudaGraphNode_t kernel_node = nullptr;
  std::size_t kernel_node_count = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    checkCuda(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
    if (type == cudaGraphNodeTypeKernel) {
      kernel_node = node;
      ++kernel_node_count;
    }
  }
  if (kernel_node_count != 1) {
    throw std::runtime_error("dispatch capture expected exactly one kernel node, "
                             "observed " +
                             std::to_string(kernel_node_count));
  }

  cudaKernelNodeParams parameters{};
  checkCuda(cudaGraphKernelNodeGetParams(kernel_node, &parameters),
            "cudaGraphKernelNodeGetParams");
  const char *kernel_name = nullptr;
  checkCuda(cudaFuncGetName(&kernel_name, parameters.func), "cudaFuncGetName");
  if (kernel_name == nullptr) {
    throw std::runtime_error("cudaFuncGetName returned a null kernel name");
  }

  DispatchObservation observation;
  observation.kernel_name = kernel_name;
  observation.path = classifyKernel(observation.kernel_name);
  observation.grid = parameters.gridDim;
  observation.block = parameters.blockDim;
  observation.shared_bytes = parameters.sharedMemBytes;

  checkCuda(cudaGraphInstantiate(&resources.executable, resources.graph, nullptr,
                                 nullptr, 0),
            "cudaGraphInstantiate");
  checkCuda(cudaGraphLaunch(resources.executable, stream), "cudaGraphLaunch");
  checkCuda(cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(dispatch graph)");
  return observation;
}

const char *dispatchPathName(DispatchPath path) {
  switch (path) {
  case DispatchPath::kUnknown:
    return "unknown";
  case DispatchPath::kScale:
    return "scale";
  case DispatchPath::kGeneral:
    return "general";
  case DispatchPath::kShared:
    return "shared";
  case DispatchPath::kRegister:
    return "register";
  case DispatchPath::kWide:
    return "wide";
  case DispatchPath::kMedium:
    return "medium";
  case DispatchPath::kSmall:
    return "small";
  case DispatchPath::kCount:
    break;
  }
  return "invalid";
}

std::string formatDispatchObservation(
    std::string_view case_name, const DispatchObservation &observation) {
  std::ostringstream output;
  output << "DISPATCH case=\"" << case_name << "\" path="
         << dispatchPathName(observation.path) << " kernel=\""
         << observation.kernel_name << "\" grid=" << observation.grid.x << 'x'
         << observation.grid.y << 'x' << observation.grid.z << " block="
         << observation.block.x << 'x' << observation.block.y << 'x'
         << observation.block.z << " shared_bytes=" << observation.shared_bytes;
  return output.str();
}

} // namespace sgblas::test
