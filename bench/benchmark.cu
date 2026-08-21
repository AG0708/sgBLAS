#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <sgblas/sgblas.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Shape {
  int m;
  int n;
  int k;
};

enum class TimingOrder {
  kSgblasFirst,
  kCublasFirst,
};

enum class OutputMode {
  kTable,
  kJsonl,
  kBoth,
};

struct Options {
  int warmups = 5;
  int repeats = 20;
  unsigned int seed = 20260711U;
  TimingOrder timing_order = TimingOrder::kSgblasFirst;
  OutputMode output_mode = OutputMode::kBoth;
  bool timing_order_was_set = false;
  std::vector<Shape> shapes;
};

struct BatchTiming {
  float total_milliseconds;
  float mean_milliseconds;
};

struct BenchmarkResult {
  Shape shape;
  BatchTiming sgblas;
  BatchTiming cublas;
  double sgblas_gflops;
  double cublas_gflops;
  double ratio;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(std::string(operation) + " failed with cuBLAS status " +
         std::to_string(static_cast<int>(status)));
  }
}

void check_sgblas(sgblasStatus_t status, std::string_view operation) {
  if (status != SGBLAS_STATUS_SUCCESS) {
    fail(std::string(operation) + " failed with sgBLAS status " +
         std::to_string(static_cast<int>(status)));
  }
}

class Stream {
public:
  Stream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  ~Stream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

class Event {
public:
  Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  ~Event() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)),
               "cudaMalloc");
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  T *get() const { return data_; }
  std::size_t size() const { return count_; }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
};

int parse_positive_int(const char *text, std::string_view name) {
  errno = 0;
  char *end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
      value > INT_MAX) {
    fail("invalid " + std::string(name) + ": " + text);
  }
  return static_cast<int>(value);
}

unsigned int parse_seed(const char *text) {
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value > UINT_MAX) {
    fail(std::string("invalid seed: ") + text);
  }
  return static_cast<unsigned int>(value);
}

TimingOrder parse_timing_order(const char *text) {
  const std::string_view value(text);
  if (value == "sgblas-first") {
    return TimingOrder::kSgblasFirst;
  }
  if (value == "cublas-first") {
    return TimingOrder::kCublasFirst;
  }
  fail(std::string("invalid timing order: ") + text +
       " (expected sgblas-first or cublas-first)");
}

OutputMode parse_output_mode(const char *text) {
  const std::string_view value(text);
  if (value == "table") {
    return OutputMode::kTable;
  }
  if (value == "jsonl") {
    return OutputMode::kJsonl;
  }
  if (value == "both") {
    return OutputMode::kBoth;
  }
  fail(std::string("invalid output mode: ") + text +
       " (expected table, jsonl, or both)");
}

const char *timing_order_name(TimingOrder order) {
  return order == TimingOrder::kSgblasFirst ? "sgblas-first"
                                             : "cublas-first";
}

bool writes_table(OutputMode mode) {
  return mode == OutputMode::kTable || mode == OutputMode::kBoth;
}

void print_usage(const char *program) {
  std::cout << "Usage: " << program
            << " [M N K] --order sgblas-first|cublas-first"
               " [--warmups N] [--repeats N] [--seed N]"
               " [--output table|jsonl|both]\n"
               "With no M/N/K, runs a default square and rectangular corpus.\n"
               "Run an equal number of processes in each timing order for a "
               "balanced comparison.\n";
}

Options parse_options(int argc, char **argv) {
  Options options;
  std::vector<int> dimensions;

  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    auto require_value = [&](std::string_view option) -> const char * {
      if (++i >= argc) {
        fail("missing value after " + std::string(option));
      }
      return argv[i];
    };

    if (argument == "--warmups") {
      options.warmups =
          parse_positive_int(require_value(argument), "warmup count");
    } else if (argument == "--repeats") {
      options.repeats =
          parse_positive_int(require_value(argument), "repeat count");
    } else if (argument == "--seed") {
      options.seed = parse_seed(require_value(argument));
    } else if (argument == "--order") {
      if (options.timing_order_was_set) {
        fail("--order may be specified only once");
      }
      options.timing_order = parse_timing_order(require_value(argument));
      options.timing_order_was_set = true;
    } else if (argument == "--output") {
      options.output_mode = parse_output_mode(require_value(argument));
    } else if (!argument.empty() && argument.front() == '-') {
      fail("unknown option: " + std::string(argument));
    } else {
      dimensions.push_back(parse_positive_int(argv[i], "dimension"));
    }
  }

  if (dimensions.empty()) {
    options.shapes = {
        {256, 256, 256},    {512, 512, 512},    {1024, 1024, 1024},
        {2048, 2048, 2048}, {4096, 4096, 4096}, {4096, 1024, 4096},
        {1024, 4096, 4096},
    };
  } else if (dimensions.size() == 3) {
    options.shapes.push_back({dimensions[0], dimensions[1], dimensions[2]});
  } else {
    fail("provide either no dimensions or exactly M N K");
  }

  if (!options.timing_order_was_set) {
    fail("--order is required; balance sgblas-first and cublas-first processes");
  }

  return options;
}

std::size_t checked_elements(int rows, int columns) {
  const auto result = static_cast<unsigned long long>(rows) *
                      static_cast<unsigned long long>(columns);
  if (result > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    fail("matrix size overflows the address space");
  }
  return static_cast<std::size_t>(result);
}

template <typename Launch>
BatchTiming time_batch(cudaStream_t stream, int repeats, Launch &&launch) {
  Event start;
  Event stop;

  check_cuda(cudaEventRecord(start.get(), stream), "cudaEventRecord(start)");
  for (int iteration = 0; iteration < repeats; ++iteration) {
    launch();
  }
  check_cuda(cudaEventRecord(stop.get(), stream), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");

  float total_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&total_milliseconds, start.get(), stop.get()),
             "cudaEventElapsedTime");
  return {total_milliseconds,
          total_milliseconds / static_cast<float>(repeats)};
}

double gflops(const Shape &shape, float milliseconds) {
  const double operations = 2.0 * static_cast<double>(shape.m) *
                            static_cast<double>(shape.n) *
                            static_cast<double>(shape.k);
  return operations / (static_cast<double>(milliseconds) * 1.0e6);
}

BenchmarkResult benchmark_shape(const Shape &shape, const Options &options,
                                std::mt19937 &generator, cudaStream_t stream,
                                sgblasHandle_t sgblas, cublasHandle_t cublas) {
  const std::size_t a_count = checked_elements(shape.m, shape.k);
  const std::size_t b_count = checked_elements(shape.k, shape.n);
  const std::size_t c_count = checked_elements(shape.m, shape.n);

  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  std::vector<float> host_a(a_count);
  std::vector<float> host_b(b_count);
  std::generate(host_a.begin(), host_a.end(),
                [&] { return distribution(generator); });
  std::generate(host_b.begin(), host_b.end(),
                [&] { return distribution(generator); });

  DeviceBuffer<float> device_a(a_count);
  DeviceBuffer<float> device_b(b_count);
  DeviceBuffer<float> sgblas_c(c_count);
  DeviceBuffer<float> cublas_c(c_count);

  check_cuda(cudaMemcpyAsync(device_a.get(), host_a.data(),
                             a_count * sizeof(float), cudaMemcpyHostToDevice,
                             stream),
             "cudaMemcpyAsync(A)");
  check_cuda(cudaMemcpyAsync(device_b.get(), host_b.data(),
                             b_count * sizeof(float), cudaMemcpyHostToDevice,
                             stream),
             "cudaMemcpyAsync(B)");
  check_cuda(
      cudaMemsetAsync(sgblas_c.get(), 0, c_count * sizeof(float), stream),
      "cudaMemsetAsync(sgBLAS C)");
  check_cuda(
      cudaMemsetAsync(cublas_c.get(), 0, c_count * sizeof(float), stream),
      "cudaMemsetAsync(cuBLAS C)");

  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;

  auto launch_sgblas = [&] {
    check_sgblas(sgblasSgemm(sgblas, SGBLAS_OP_N, SGBLAS_OP_N, shape.m, shape.n,
                             shape.k, &alpha, device_a.get(), shape.m,
                             device_b.get(), shape.k, &beta, sgblas_c.get(),
                             shape.m),
                 "sgblasSgemm");
  };

  auto launch_cublas = [&] {
    check_cublas(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, shape.m,
                              shape.n, shape.k, &alpha, device_a.get(),
                              CUDA_R_32F, shape.m, device_b.get(), CUDA_R_32F,
                              shape.k, &beta, cublas_c.get(), CUDA_R_32F,
                              shape.m, CUBLAS_COMPUTE_32F_PEDANTIC,
                              CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx(strict FP32)");
  };

  for (int iteration = 0; iteration < options.warmups; ++iteration) {
    if ((iteration & 1) == 0) {
      launch_sgblas();
      launch_cublas();
    } else {
      launch_cublas();
      launch_sgblas();
    }
  }
  check_cuda(cudaStreamSynchronize(stream), "warmup synchronization");

  BatchTiming sgblas_timing{};
  BatchTiming cublas_timing{};
  // Keep each library in its own event window. The process-level order is
  // explicit so a runner can balance AB and BA processes for every shape.
  if (options.timing_order == TimingOrder::kSgblasFirst) {
    sgblas_timing = time_batch(stream, options.repeats, launch_sgblas);
    cublas_timing = time_batch(stream, options.repeats, launch_cublas);
  } else {
    cublas_timing = time_batch(stream, options.repeats, launch_cublas);
    sgblas_timing = time_batch(stream, options.repeats, launch_sgblas);
  }

  const double sgblas_gflops = gflops(shape, sgblas_timing.mean_milliseconds);
  const double cublas_gflops = gflops(shape, cublas_timing.mean_milliseconds);
  const double ratio = sgblas_gflops / cublas_gflops;

  return {shape,          sgblas_timing, cublas_timing, sgblas_gflops,
          cublas_gflops, ratio};
}

std::string json_string(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
    case '"':
      output << "\\\"";
      break;
    case '\\':
      output << "\\\\";
      break;
    case '\b':
      output << "\\b";
      break;
    case '\f':
      output << "\\f";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (character < 0x20U) {
        output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<unsigned int>(character) << std::dec
               << std::setfill(' ');
      } else {
        output << static_cast<char>(character);
      }
    }
  }
  output << '"';
  return output.str();
}

template <typename T> std::string full_precision(T value) {
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
  return output.str();
}

std::string format_cuda_version(int version) {
  return std::to_string(version / 1000) + '.' +
         std::to_string((version % 1000) / 10);
}

std::string format_cublas_version(int version) {
  return std::to_string(version / 10000) + '.' +
         std::to_string((version % 10000) / 100) + '.' +
         std::to_string(version % 100);
}

void print_table_result(const BenchmarkResult &result) {
  std::cout << std::setw(6) << result.shape.m << std::setw(7) << result.shape.n
            << std::setw(7) << result.shape.k << std::setw(13) << std::fixed
            << std::setprecision(3) << result.sgblas.mean_milliseconds
            << std::setw(15) << std::setprecision(1) << result.sgblas_gflops
            << std::setw(13) << std::setprecision(3)
            << result.cublas.mean_milliseconds << std::setw(15)
            << std::setprecision(1) << result.cublas_gflops << std::setw(11)
            << std::setprecision(3) << result.ratio << '\n';
}

void print_json_metadata(const Options &options,
                         const cudaDeviceProp &properties, int device,
                         std::string_view pci_bus_id, int cuda_driver_version,
                         int cuda_runtime_version, int cublas_version, int argc,
                         char **argv) {
  const char *tf32_override = std::getenv("NVIDIA_TF32_OVERRIDE");
  std::ostringstream output;
  output << "{\"record_type\":\"metadata\",\"schema_version\":1"
         << ",\"benchmark\":{\"timed_order\":"
         << json_string(timing_order_name(options.timing_order))
         << ",\"warmup_order\":\"alternating-per-launch\""
         << ",\"warmups_per_implementation\":" << options.warmups
         << ",\"timed_repeats_per_implementation\":" << options.repeats
         << ",\"seed\":" << options.seed
         << ",\"cache_policy\":\"same-buffer-steady-state\""
         << ",\"stream\":\"shared-nonblocking\""
         << ",\"sgblas_math_mode\":\"SGBLAS_MATH_FP32\""
         << ",\"cublas_math_mode\":\"CUBLAS_PEDANTIC_MATH\""
         << ",\"cublas_compute_type\":\"CUBLAS_COMPUTE_32F_PEDANTIC\""
         << ",\"cublas_algorithm\":\"CUBLAS_GEMM_DEFAULT\""
         << ",\"nvidia_tf32_override\":"
         << (tf32_override == nullptr ? "null" : json_string(tf32_override))
         << "}"
         << ",\"cuda\":{\"driver_version_raw\":" << cuda_driver_version
         << ",\"driver_version\":"
         << json_string(format_cuda_version(cuda_driver_version))
         << ",\"runtime_version_raw\":" << cuda_runtime_version
         << ",\"runtime_version\":"
         << json_string(format_cuda_version(cuda_runtime_version)) << "}"
         << ",\"cublas\":{\"version_raw\":" << cublas_version
         << ",\"version\":"
         << json_string(format_cublas_version(cublas_version)) << "}"
         << ",\"device\":{\"ordinal\":" << device
         << ",\"name\":" << json_string(properties.name)
         << ",\"pci_bus_id\":" << json_string(pci_bus_id)
         << ",\"compute_capability_major\":" << properties.major
         << ",\"compute_capability_minor\":" << properties.minor
         << ",\"multiprocessor_count\":" << properties.multiProcessorCount
         << ",\"total_global_memory_bytes\":"
         << static_cast<unsigned long long>(properties.totalGlobalMem)
         << ",\"l2_cache_bytes\":" << properties.l2CacheSize
         << ",\"shared_memory_per_block_bytes\":"
         << static_cast<unsigned long long>(properties.sharedMemPerBlock)
         << ",\"shared_memory_per_block_optin_bytes\":"
         << static_cast<unsigned long long>(properties.sharedMemPerBlockOptin)
         << ",\"shared_memory_per_multiprocessor_bytes\":"
         << static_cast<unsigned long long>(properties.sharedMemPerMultiprocessor)
         << ",\"registers_per_multiprocessor\":"
         << properties.regsPerMultiprocessor
         << ",\"warp_size\":" << properties.warpSize
         << ",\"max_threads_per_multiprocessor\":"
         << properties.maxThreadsPerMultiProcessor
         << ",\"reported_core_clock_khz\":" << properties.clockRate
         << ",\"reported_memory_clock_khz\":" << properties.memoryClockRate
         << ",\"memory_bus_width_bits\":" << properties.memoryBusWidth << "}"
         << ",\"argv\":[";
  for (int index = 0; index < argc; ++index) {
    if (index != 0) {
      output << ',';
    }
    output << json_string(argv[index]);
  }
  output << "]}";
  std::cout << output.str() << '\n';
}

void print_json_result(const BenchmarkResult &result, const Options &options) {
  std::cout
      << "{\"record_type\":\"result\",\"schema_version\":1"
      << ",\"timed_order\":"
      << json_string(timing_order_name(options.timing_order))
      << ",\"warmups_per_implementation\":" << options.warmups
      << ",\"timed_repeats_per_implementation\":" << options.repeats
      << ",\"m\":" << result.shape.m << ",\"n\":" << result.shape.n
      << ",\"k\":" << result.shape.k << ",\"sgblas_total_ms\":"
      << full_precision(result.sgblas.total_milliseconds)
      << ",\"sgblas_mean_ms\":"
      << full_precision(result.sgblas.mean_milliseconds)
      << ",\"sgblas_gflops\":" << full_precision(result.sgblas_gflops)
      << ",\"cublas_total_ms\":"
      << full_precision(result.cublas.total_milliseconds)
      << ",\"cublas_mean_ms\":"
      << full_precision(result.cublas.mean_milliseconds)
      << ",\"cublas_gflops\":" << full_precision(result.cublas_gflops)
      << ",\"ratio\":" << full_precision(result.ratio) << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    int cuda_driver_version = 0;
    int cuda_runtime_version = 0;
    check_cuda(cudaDriverGetVersion(&cuda_driver_version),
               "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&cuda_runtime_version),
               "cudaRuntimeGetVersion");
    char pci_bus_id[32] = {};
    check_cuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device),
               "cudaDeviceGetPCIBusId");

    Stream stream;
    sgblasHandle_t sgblas = nullptr;
    cublasHandle_t cublas = nullptr;
    check_sgblas(sgblasCreate(&sgblas), "sgblasCreate");
    check_cublas(cublasCreate(&cublas), "cublasCreate");

    try {
      check_sgblas(
          sgblasSetStream(sgblas, reinterpret_cast<void *>(stream.get())),
          "sgblasSetStream");
      check_sgblas(sgblasSetMathMode(sgblas, SGBLAS_MATH_FP32),
                   "sgblasSetMathMode(FP32)");
      check_cublas(cublasSetStream(cublas, stream.get()), "cublasSetStream");
      check_cublas(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH),
                   "cublasSetMathMode(PEDANTIC)");

      int cublas_version = 0;
      check_cublas(cublasGetVersion(cublas, &cublas_version),
                   "cublasGetVersion");

      if (writes_table(options.output_mode)) {
        const char *tf32_override = std::getenv("NVIDIA_TF32_OVERRIDE");
        std::cout
            << "GPU: " << properties.name << " (SM " << properties.major << '.'
            << properties.minor << ")\n"
            << "PCI bus ID: " << pci_bus_id << '\n'
            << "CUDA driver/runtime: "
            << format_cuda_version(cuda_driver_version) << " ("
            << cuda_driver_version << ") / "
            << format_cuda_version(cuda_runtime_version) << " ("
            << cuda_runtime_version << ")\n"
            << "cuBLAS version: " << format_cublas_version(cublas_version)
            << " (" << cublas_version << ")\n"
            << "Device resources: " << properties.multiProcessorCount
            << " SMs, "
            << static_cast<unsigned long long>(properties.totalGlobalMem)
            << " global bytes, " << properties.l2CacheSize << " L2 bytes\n"
            << "Reported device-property clocks: " << properties.clockRate
            << " kHz core, " << properties.memoryClockRate << " kHz memory\n"
            << "Math: strict FP32; column-major NN SGEMM; alpha=1 beta=0\n"
            << "cuBLAS contract: CUBLAS_PEDANTIC_MATH, "
               "CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT\n"
            << "NVIDIA_TF32_OVERRIDE: "
            << (tf32_override == nullptr ? "<unset>" : tf32_override) << '\n'
            << "Warmups per implementation: " << options.warmups
            << ", timed repeats per implementation: " << options.repeats << '\n'
            << "Timed order: " << timing_order_name(options.timing_order) << '\n'
            << "Warmup order: alternating-per-launch\n"
            << "Stream: shared nonblocking CUDA stream\n"
            << "Cache policy: same-buffer-steady-state\n\n"
            << std::setw(6) << "M" << std::setw(7) << "N" << std::setw(7)
            << "K" << std::setw(13) << "sgBLAS ms" << std::setw(15)
            << "sgBLAS GF/s" << std::setw(13) << "cuBLAS ms"
            << std::setw(15) << "cuBLAS GF/s" << std::setw(11) << "ratio"
            << '\n';
      }

      if (options.output_mode == OutputMode::kJsonl) {
        print_json_metadata(options, properties, device, pci_bus_id,
                            cuda_driver_version, cuda_runtime_version,
                            cublas_version, argc, argv);
      }

      std::mt19937 generator(options.seed);
      std::vector<BenchmarkResult> results;
      results.reserve(options.shapes.size());
      for (const Shape &shape : options.shapes) {
        BenchmarkResult result = benchmark_shape(
            shape, options, generator, stream.get(), sgblas, cublas);
        if (writes_table(options.output_mode)) {
          print_table_result(result);
        }
        if (options.output_mode == OutputMode::kJsonl) {
          print_json_result(result, options);
        }
        results.push_back(result);
      }
      if (writes_table(options.output_mode)) {
        std::cout << "\nratio = sgBLAS GF/s / cuBLAS GF/s\n";
      }
      if (options.output_mode == OutputMode::kBoth) {
        std::cout << "\nMachine-readable JSONL:\n";
        print_json_metadata(options, properties, device, pci_bus_id,
                            cuda_driver_version, cuda_runtime_version,
                            cublas_version, argc, argv);
        for (const BenchmarkResult &result : results) {
          print_json_result(result, options);
        }
      }
    } catch (...) {
      cublasDestroy(cublas);
      sgblasDestroy(sgblas);
      throw;
    }

    check_cublas(cublasDestroy(cublas), "cublasDestroy");
    check_sgblas(sgblasDestroy(sgblas), "sgblasDestroy");
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "benchmark error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
