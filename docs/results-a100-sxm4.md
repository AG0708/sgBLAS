# Archived A100-SXM4 strict-FP32 tuning snapshot

> **Not release evidence.** These measurements were produced before the source
> tree had an immutable Git commit, and the four sanitizer runs did not cover
> every kernel in the final shape-dispatched portfolio. The arithmetic below is
> reproducible from retained local logs, but the logs cannot prove that the
> current source produced the binaries. Do not use these numbers as a resume or
> public release claim. A fresh commit-bound run supersedes this document.

## Scope

This result covers column-major `NN` SGEMM with FP32 inputs, FP32 FMA
accumulation, FP32 output, `alpha=1`, and `beta=0`. It is not a TF32 or Tensor
Core comparison.

The reported values are medians from five independent benchmark processes.
Each process used 10 warmup launches and 100 timed launches per implementation
on the same nonblocking CUDA stream, with `NVIDIA_TF32_OVERRIDE=0`. These are
repeated-buffer, steady-state CUDA-event timings—not cold-cache or end-to-end
measurements.

## Target

| Field | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| Compute capability | 8.0 (`sm_80`) |
| Driver | 580.126.16 |
| Power limit | 400 W |
| Maximum SM clock | 1410 MHz |
| Maximum memory clock | 1593 MHz |
| CUDA compiler | 12.4.131 |
| cuBLAS | 12.4.5 (`120405`) |
| Host | Ubuntu 22.04, Linux 6.8.0-100 |

## Winning SM80 kernel portfolio

- Main CTA tile: `128x64x32`, 256 threads, `4x8` outputs per thread
- Thread map: 32 logical rows by 8 logical columns, giving warp-wide
  contiguous C stores and warp-broadcast B reads
- Two-stage A/B shared-memory pipeline using 16-byte `cp.async` copies
- Dynamic shared memory: 49,152 bytes per CTA. The kernel requests the maximum
  shared-memory carveout as a driver preference, not a residency guarantee; the
  resource footprint permits at most two CTAs per SM on A100.
- Registers: 126 per thread; register spills: zero
- Main dispatch: aligned full tiles with `K >= 64` and at least 128 equivalent
  `128x128` CTAs
- Underfilled fallback: the zero-spill `128x128x32` async kernel remains faster
  for the 1024-cubed bucket
- General fallbacks: the 122-register synchronous kernel handles edge tiles,
  the shared-memory kernel handles smaller shapes, and the semantic kernel
  handles transpose modes

## Five-process median

| M | N | K | sgBLAS ms | sgBLAS GFLOP/s | cuBLAS ms | cuBLAS GFLOP/s | Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 256 | 0.024 | 1,380.6 | 0.012 | 2,745.7 | 0.502 |
| 512 | 512 | 512 | 0.074 | 3,645.8 | 0.028 | 9,748.2 | 0.374 |
| 1024 | 1024 | 1024 | 0.212 | 10,121.6 | 0.130 | 16,542.2 | 0.612 |
| 2048 | 2048 | 2048 | 1.029 | 16,689.9 | 0.972 | 17,669.0 | **0.945** |
| 4096 | 4096 | 4096 | 7.727 | **17,786.7** | 7.225 | 19,023.6 | **0.935** |
| 4096 | 1024 | 4096 | 2.051 | 16,749.8 | 1.845 | 18,627.0 | **0.899** |
| 1024 | 4096 | 4096 | 2.052 | 16,748.0 | 1.974 | 17,405.3 | **0.962** |

The geometric mean of the ratios for the final four large shapes is **0.9350**,
up from **0.7926** for the wide async kernel and **0.6891** for the synchronous
kernel in same-pod runs. At 4096 cubed, the five candidate runs span 17,779.9
to 17,801.0 GFLOP/s with a 17,786.7 GFLOP/s median.

Throughput uses conventional GEMM algorithmic work, `2MNK/t`, with one FMA
counted as two operations. NVIDIA's
[A100 80GB datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf)
specifies 19.5 TFLOP/s nominal FP32 peak for the SXM model, so 17.7867 TFLOP/s
is 91.2% of nominal peak. The 93.5% figure
is only the measured sgBLAS/cuBLAS ratio for this archived benchmark; it is not
percent of peak or a universal cuBLAS comparison.

The optional `8192x8192x8192` sustained-throughput case used five independent
processes, 10 warmups, and 50 repeats. In an alternating comparison with the
previous winner, it reached a **17,750.8 GFLOP/s** median in 61.942 ms versus
cuBLAS at 19,153.3 GFLOP/s in 57.406 ms: **0.927** of strict-FP32 cuBLAS. The
five sgBLAS runs ranged from 17,684.2 to 17,873.8 GFLOP/s.

## Optimization progression at 4096 cubed

| Kernel | sgBLAS GFLOP/s | cuBLAS ratio |
|---|---:|---:|
| One output per thread | 563.6 | 0.030 |
| 32x32 shared-memory tile | 6,388.6 | 0.337 |
| 128x128x8 register tile | 14,109.8 | 0.741 |
| Vectorized 128x128x16 | 14,987.0 | 0.787 |
| Vectorized 128x128x32 | 15,228.7 | 0.801 |
| 128x128 SM80 two-stage `cp.async` | 16,540.7 | 0.869 |
| 128x64 SM80 two-stage `cp.async` | **17,786.7** | **0.935** |

## Correctness and diagnostics

The CUDA correctness suite covers every transpose pair, padded leading
dimensions, odd dimensions, nontrivial alpha/beta values, an alpha-zero path,
a large `NN` case with M/N/K tails through the register kernel, a fully aligned
`768x768x64` case that activates the wide asynchronous pipeline, and an aligned
`1024x2048x64` case that activates the 128x64 pipeline.

An earlier six-case, pre-portfolio binary produced clean logs for:

- Compute Sanitizer `memcheck`: zero errors
- Compute Sanitizer `racecheck`: zero errors and zero warnings
- Compute Sanitizer `initcheck`: zero errors
- Compute Sanitizer `synccheck`: zero errors

Those logs do not exercise every kernel in the portfolio that produced the
table, so they are historical diagnostics rather than evidence for the final
dispatch. This gap is one reason the snapshot is barred from release claims.

Nsight Compute hardware performance counters were unavailable on this RunPod
host (`ERR_NVGPUCTRPERM`). Compiler resource reports, SASS inspection, CUDA
events, correctness tests, and sanitizer results remain available. No profiler
counter claim is made.
