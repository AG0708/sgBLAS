# sgBLAS

[![CI](https://github.com/AG0708/sgBLAS/actions/workflows/ci.yml/badge.svg)](https://github.com/AG0708/sgBLAS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/AG0708/sgBLAS)](https://github.com/AG0708/sgBLAS/releases/latest)
[![License](https://img.shields.io/github/license/AG0708/sgBLAS)](LICENSE)

sgBLAS is a from-scratch CUDA SGEMM project: learn the machinery behind a high-performance BLAS implementation, then push a deliberately narrow kernel family as close as practical to NVIDIA cuBLAS on the same GPU using the same declared operand, output, and accumulation precision.

This is not a claim to replace all of cuBLAS. cuBLAS covers many datatypes, layouts, shapes, architectures, batched operations, and numerical modes with a large internal kernel portfolio. The first defensible goal here is much smaller:

- dense, single-GPU, column-major SGEMM;
- FP32 inputs, FP32 accumulation, and FP32 output;
- `C = alpha * op(A) * op(B) + beta * C`;
- `NN`, `NT`, `TN`, and `TT`, arbitrary legal dimensions and leading dimensions;
- asynchronous, stream-ordered execution;
- a correct general kernel, followed by an optimized `NN` path;
- performance compared only with cuBLAS strict FP32, never with TF32.

The current public C API is in `include/sgblas/sgblas.h`. The design and measurement contracts are in [docs/design.md](docs/design.md) and [docs/benchmarking.md](docs/benchmarking.md).

## What success means

Correctness is a gate, not a score. Every supported transpose mode, tail shape, legal leading dimension, `alpha`/`beta` case, and non-default CUDA stream must pass before a kernel enters the performance table.

In this project, “strict FP32” means FP32 inputs and outputs, FP32
multiplication and accumulation, cuBLAS
`CUBLAS_COMPUTE_32F_PEDANTIC`/`CUBLAS_PEDANTIC_MATH`, and benchmark processes
launched with `NVIDIA_TF32_OVERRIDE=0`. It excludes TF32 input conversion; it
does not promise bitwise-identical results or an identical reduction order.
The ratio is against this measured pedantic baseline, not cuBLAS's fastest
available math mode.

Performance is reported as

```text
ratio = sgBLAS strict-FP32 GFLOP/s / cuBLAS strict-FP32 GFLOP/s
```

on an identified NVIDIA GPU. The initial large-`NN` milestones are:

| Level | Geometric-mean ratio on the target corpus | Interpretation |
|---|---:|---|
| Baseline | 0.50 | Tiling and memory reuse are working |
| Competitive | 0.75 | Register blocking and scheduling are effective |
| Stretch | 0.85 | Close to cuBLAS for the deliberately narrow target |

These are project targets, not promises of universal performance. Results outside the declared GPU, shape corpus, transpose mode, cache policy, and math mode do not support a general “percent of cuBLAS” claim.

### Release-result status

On one NVIDIA A100-SXM4-80GB, the v0.1.0 kernel portfolio reached a
six-process median of **17.736 TFLOP/s** at `4096x4096x4096`. Across the four
declared large `NN` shapes, the geometric mean of the per-shape median
throughput ratios was **94.08%** versus cuBLAS configured with
`CUBLAS_COMPUTE_32F_PEDANTIC`, with TF32 disabled.

Each scored standard-corpus process used 10 warmups and 100 CUDA-event-timed
launches per implementation on one shared nonblocking stream under a
same-buffer, steady-state cache policy. All 24 checked correctness cases and
all seven required dispatch paths passed. Compute Sanitizer `memcheck`,
`racecheck`, `initcheck`, and `synccheck` were clean.

The fail-closed verifier accepted all 45 checksummed evidence artifacts bound
to commit
[`b26974f`](https://github.com/AG0708/sgBLAS/commit/b26974f9d25f7a904d2141b15cdde2f6663e106d).
The [v0.1.0 release](https://github.com/AG0708/sgBLAS/releases/tag/v0.1.0)
publishes the raw logs, tested-binary hashes, source digest, build
configuration, machine telemetry, deterministic archives, checksums, and SPDX
SBOM. See [the full A100 scorecard](docs/results-a100-sxm4.md).

These measurements apply only to this GPU, corpus, math contract, and protocol;
they are not a claim of universal cuBLAS parity.

## Platform reality

The repository can be configured and its host-side API tests can run on this
Mac. CUDA kernels cannot be compiled or executed natively for macOS: NVIDIA no
longer supports CUDA development on macOS, and current CUDA installation
requirements call for a CUDA-capable NVIDIA GPU on supported Linux. A Linux
CUDA container can still compile the source on this Mac, but it cannot execute
or benchmark a kernel without an NVIDIA device. See NVIDIA's
[macOS notice](https://developer.nvidia.com/nvidia-cuda-toolkit-12_9_0-developer-tools-mac-hosts)
and [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

### Host-only development on macOS

```bash
cmake --preset host-debug
cmake --build --preset host-debug --target sgblas_host_tests
ctest --preset host-debug
```

This validates the portable API layer and argument handling. It does not validate CUDA compilation, device execution, numerical results, or performance.

On an ARM64 Mac with Docker Desktop, the CUDA sources can also be compile- and
link-checked without a GPU:

```bash
SGBLAS_CUDA_ARCHITECTURE=80 \
SGBLAS_EXPERIMENTAL_SM80_ASYNC=ON \
SGBLAS_EXPERIMENTAL_SM80_MEDIUM=ON \
SGBLAS_EXPERIMENTAL_SM80_SMALL=ON \
./tools/compile_cuda_container.sh
```

That uses the version-tagged CUDA development image in `infra/runpod/Dockerfile`.
The evidence manifest records the resolved image and toolchain separately. A
successful container build still does not execute a kernel; device tests remain
mandatory on the RunPod target.

### Build and run on an NVIDIA Linux machine

Capture the machine identity before comparing results:

```bash
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,compute_cap,memory.total,power.limit,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu --format=csv
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE
nvcc --version
cmake --version
uname -a
```

Then configure for the installed GPU, build, and test:

```bash
cmake --preset cuda-release \
  -DSGBLAS_EXPERIMENTAL_SM80_ASYNC=ON \
  -DSGBLAS_EXPERIMENTAL_SM80_MEDIUM=ON \
  -DSGBLAS_EXPERIMENTAL_SM80_SMALL=ON
cmake --build --preset cuda-release --target sgblas_cuda_correctness sgblas_benchmark -j
ctest --preset cuda-release
```

Run the supplied strict-FP32 comparison corpus or one shape:

```bash
NVIDIA_TF32_OVERRIDE=0 ./build/cuda-release/sgblas_benchmark --order sgblas-first --warmups 10 --repeats 100
NVIDIA_TF32_OVERRIDE=0 ./build/cuda-release/sgblas_benchmark 4096 4096 4096 --order cublas-first --warmups 10 --repeats 100
```

The preset uses `CMAKE_CUDA_ARCHITECTURES=native`. For a controlled build farm, configure an explicit architecture instead of silently reusing a binary built for another GPU.

For a reproducible same-pod A/B campaign between the wide and hybrid SM80
portfolios, run:

```bash
./tools/run_a100_tuning.py \
  --runs 6 --warmups 10 --repeats 100 \
  --include-8192 --sanitizers
```

The runner builds isolated variants, rejects compiler-reported spills, gates on
correctness, alternates process order, records a deterministic source hash and
machine probe, summarizes medians, and stores raw artifacts under
`results/tuning/`.

Treat `state=complete` as necessary but not sufficient. After copying the whole
run directory off the GPU host, verify it against the same clean source commit:

```bash
python3 tools/verify_evidence.py /absolute/path/to/run-root --source "$PWD"
```

Only a successful fail-closed verification can promote a result into the
release or resume evidence ledger.

## Optimization path

The project advances one measured step at a time:

1. scalar correctness kernel;
2. coalesced global access;
3. shared-memory CTA tiling;
4. bank-conflict-free shared layouts;
5. aligned vectorized loads and stores;
6. per-thread register tiles and an unrolled K loop;
7. double or multistage copy/compute pipelining;
8. shape-aware launch configuration, rasterization, and dispatch;
9. architecture-specific asynchronous copy and scheduling;
10. TF32/Tensor Core kernels as a separately named and separately scored mode.

NVIDIA's current [CUTLASS efficient GEMM description](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) is the architectural reference, not a dependency. The authoritative low-level references are the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/), and [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/).

## Repository targets

| Target | Purpose | Runs on this Mac? |
|---|---|---:|
| `sgblas` | Library | Host stub only |
| `sgblas_host_tests` | API and validation tests | Yes |
| `sgblas_cuda_correctness` | Device correctness and stream tests | No |
| `sgblas_benchmark` | Strict-FP32 sgBLAS/cuBLAS comparison | No |
