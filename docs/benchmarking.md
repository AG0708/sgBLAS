# Benchmarking contract

The benchmark exists to answer a narrow question reproducibly: on one identified NVIDIA GPU, how fast is sgBLAS's FP32 `NN` SGEMM relative to cuBLAS performing the same strict-FP32 operation? “Strict FP32” is project shorthand for FP32 inputs, FP32 multiply, and FP32 accumulation with TF32 disabled; it does not promise bitwise-identical output or an identical reduction order.

## Fairness rules

The primary lane fixes all of the following:

- column-major `NN` SGEMM;
- FP32 A, B, and C;
- FP32 multiply and FP32 accumulation;
- `alpha = 1`, `beta = 0`;
- identical M/N/K, leading dimensions, device pointers, stream, and cache treatment;
- sgBLAS `SGBLAS_MATH_FP32`;
- cuBLAS `CUBLAS_PEDANTIC_MATH` with an explicit
  `CUBLAS_COMPUTE_32F_PEDANTIC` `cublasGemmEx` call;
- `NVIDIA_TF32_OVERRIDE=0` as defense in depth.

TF32 is a future, separate lane. It must compare `SGBLAS_MATH_TF32` with cuBLAS `CUBLAS_COMPUTE_32F_FAST_TF32` or the explicitly documented TF32 math mode, and its results must be labelled `TF32`. NVIDIA documents these modes separately in the [cuBLAS manual](https://docs.nvidia.com/cuda/cublas/).

Do not compare against a cuBLAS result whose math mode, compute type, or environment is unknown. Do not use `-use_fast_math` in the primary sgBLAS build. Do not include allocation, initialization, host/device copies, context creation, handle creation, or one-time dispatch setup in kernel throughput; report end-to-end latency separately if it later matters.

## Required runner probe

Every saved result must include:

- UTC timestamp and git commit;
- OS and kernel;
- GPU name, PCI bus ID, compute capability, SM count, and total memory (omit stable hardware UUIDs from public artifacts);
- driver, CUDA runtime, CUDA toolkit, and cuBLAS versions;
- configured CUDA architecture target;
- shared memory per SM and per block, opt-in shared-memory limit, register count per SM, warp size, and maximum threads per SM;
- persistence mode, MIG/MPS state, power limit, performance state, current/application SM and memory clocks, and temperature;
- whether clocks were locked and whether another process used the GPU;
- exact benchmark command, seed, warmup count, repeat count, and `NVIDIA_TF32_OVERRIDE` value.

Start with:

```bash
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,compute_cap,memory.total,power.limit,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu --format=csv
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE
nvcc --version
cmake --version
uname -a
git rev-parse HEAD
```

The CUDA runner should additionally print `cudaGetDeviceProperties` and relevant `cudaDeviceGetAttribute` values because `nvidia-smi` does not expose every kernel resource. NVIDIA lists the required CUDA-capable GPU and supported Linux environment in the [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

## Build and execute

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release --target sgblas_cuda_correctness sgblas_benchmark -j
ctest --preset cuda-release

NVIDIA_TF32_OVERRIDE=0 ./build/cuda-release/sgblas_benchmark --help
NVIDIA_TF32_OVERRIDE=0 ./build/cuda-release/sgblas_benchmark --order sgblas-first --warmups 10 --repeats 100
NVIDIA_TF32_OVERRIDE=0 ./build/cuda-release/sgblas_benchmark 4096 4096 4096 --order cublas-first --warmups 10 --repeats 100
```

The A100 tuning runner automates the same contract for isolated wide/hybrid
builds and retains the raw evidence:

```bash
./tools/run_a100_tuning.py \
  --runs 6 --warmups 10 --repeats 100 \
  --include-8192 --sanitizers
```

The no-dimension form runs the checked-in square and rectangular corpus. Three positional dimensions run one `M N K` case. `--seed` makes initialization reproducible.

## Timing protocol

1. Use a dedicated, nonblocking CUDA stream for both libraries.
2. Initialize both libraries from identical A/B data and independent C buffers.
3. Execute at least 10 unmeasured warmups for each implementation.
4. Record CUDA events in the measured stream, synchronize the stop event, and divide a batch time by its repeat count.
5. Run the entire executable at least five times. Use the median process-level GFLOP/s for the scorecard and retain the minimum, median, and maximum to expose clock or thermal instability.
6. Alternate which library is timed first across shapes or process runs.
7. Keep the GPU idle except for the benchmark. Record throttling, temperature, and clocks.
8. Apply the same cache policy to both implementations. Label repeated same-buffer measurements `warm-cache`. For a `cold/L2-safe` result, rotate workspaces whose aggregate live footprint exceeds L2; do not flush only one implementation.

CUDA events use the GPU clock and avoid asynchronous host-timer mistakes; NVIDIA gives the canonical method in [Using CUDA GPU Timers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). NVIDIA's current [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html) likewise uses warmup iterations and multiple workspaces to avoid accidental last-level-cache residency. For profiler runs, document Nsight Compute's cache and clock controls because replay can change both; see the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

For each implementation and shape:

```text
GFLOP/s = 2 * M * N * K / elapsed_seconds / 1e9
ratio   = sgBLAS_GFLOP/s / cuBLAS_GFLOP/s
```

This normalization intentionally counts one multiply and one add per inner-product term. It excludes the small epilogue operation count so results remain comparable with conventional GEMM reporting.

## Shape corpus

### Primary optimized-`NN` scorecard

The checked-in default corpus is:

```text
256x256x256
512x512x512
1024x1024x1024
2048x2048x2048
4096x4096x4096
4096x1024x4096
1024x4096x4096
```

Report every row, but compute the headline large-shape geometric mean from the final four rows. The smaller rows reveal launch and edge overhead; they should not dominate the throughput headline.

When memory permits, also record `8192x8192x8192`. Real application shapes may be added as a separate corpus, but the corpus must be frozen before comparing commits.

### Correctness-only coverage

Performance specialization does not reduce the API contract. Device correctness tests must include:

- all `NN`, `NT`, `TN`, and `TT` modes;
- prime and off-by-one dimensions around tile boundaries;
- `m`, `n`, or `k` equal to zero;
- padded legal leading dimensions;
- nontrivial `alpha` and `beta`, including `alpha=0` and `beta=0`;
- a non-default CUDA stream;
- values that reveal an illegal read of C when `beta=0`.

## Scorecard

| Field | Gate or reported value |
|---|---|
| Correctness suite | 100% pass; otherwise no performance score |
| Compute contract | Strict FP32, explicitly printed |
| Target corpus | Per-shape milliseconds, GFLOP/s, and ratio |
| Headline score | Geometric mean of large-shape ratios |
| Baseline milestone | `>= 0.50` geometric mean |
| Competitive milestone | `>= 0.75` geometric mean |
| Stretch milestone | `>= 0.85` geometric mean |
| Stability | Five process runs with min/median/max retained |
| Regressions | No supported correctness case may fall back incorrectly or fail |

Nsight Compute is diagnostic evidence, not the score itself. Use its roofline, memory workload, occupancy, scheduler, source/SASS, and warp-stall sections to explain changes in DRAM/L2 traffic, shared-memory conflicts, register spills, instruction mix, eligible warps, and FP32-pipe utilization. NVIDIA documents these facilities in the [Nsight Compute guide](https://docs.nvidia.com/nsight-compute/NsightCompute/).
