# A100-SXM4 strict-FP32 scorecard: v0.1.0

This is the commit-bound performance and correctness record for
[sgBLAS v0.1.0](https://github.com/AG0708/sgBLAS/releases/tag/v0.1.0).
The fail-closed verifier accepted the complete 45-artifact evidence bundle for
commit
[`b26974f`](https://github.com/AG0708/sgBLAS/commit/b26974f9d25f7a904d2141b15cdde2f6663e106d).

## Headline

On one NVIDIA A100-SXM4-80GB, sgBLAS reached a six-process median of
**17.736 TFLOP/s** at `4096x4096x4096`. Across the four declared large `NN`
shapes, the geometric mean of the per-shape median throughput ratios was
**94.08%** versus pedantic cuBLAS with TF32 disabled.

This is a deliberately scoped result, not a claim of universal cuBLAS parity.
It covers the GPU, shapes, layout, math contract, and protocol below.

## Contract and protocol

- Column-major `NN` SGEMM with FP32 inputs, FP32 accumulation, FP32 output,
  `alpha=1`, and `beta=0`
- cuBLAS configured with `CUBLAS_COMPUTE_32F_PEDANTIC` and
  `CUBLAS_PEDANTIC_MATH`; benchmark processes launched with
  `NVIDIA_TF32_OVERRIDE=0`
- Six benchmark processes per variant for the seven standard shapes, plus a
  separate six-process set for the optional `8192` shape; each set is balanced
  three-and-three between `sgblas-first` and `cublas-first` timing order
- 10 warmup launches and 100 CUDA-event-timed launches per implementation;
  the optional `8192` row uses 50 timed launches
- One shared nonblocking stream, identical A/B data, independent C buffers,
  and a repeated-buffer steady-state cache policy; allocation and setup are
  outside the timed region
- Median throughput per shape; the headline score is the geometric mean of the
  four large-shape median ratios

“Strict FP32” is project shorthand for this contract. It excludes TF32 input
conversion, but it does not imply bitwise-identical results or an identical
reduction order.

## Target and provenance

| Field | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| Compute capability | 8.0 (`sm_80`) |
| Memory | 81,920 MiB |
| Driver | 580.126.16 |
| CUDA compiler | 12.8.93 |
| Container OS / host kernel | Ubuntu 24.04.3 LTS / Linux 6.8.0-100 |
| Container | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` |
| Tested commit | `b26974f9d25f7a904d2141b15cdde2f6663e106d` |
| Campaign source snapshot digest | `7941e5af45c49a945be3b4d7450965bd1c177432c81b1584725768cc7297cf3d` |
| Release source-tree digest | `fb1b178952819c2c19e31fc6046ce92743a58186e2146dfce87e6dd31b119921` |
| Campaign duration | 231.684 seconds |

The evidence manifest records the resolved container digest, tool paths, OS
digest, redacted NVIDIA system report, process list, 50 telemetry snapshots,
146 commands, CMake configuration, compiler spill reports, tested-binary
hashes, and source hashes before and after the run. The Git worktree was clean
and unchanged throughout.

## Six-process medians

| M | N | K | sgBLAS ms | sgBLAS GFLOP/s | cuBLAS ms | cuBLAS GFLOP/s | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 256 | 0.02439 | 1,375.8 | 0.01228 | 2,733.4 | 0.504 |
| 512 | 512 | 512 | 0.03295 | 8,147.2 | 0.02755 | 9,744.3 | 0.836 |
| 1024 | 1024 | 1024 | 0.14165 | 15,160.2 | 0.13013 | 16,502.7 | 0.919 |
| **1024** | **4096** | **4096** | **2.04776** | **16,779.2** | **1.98454** | **17,313.7** | **0.969** |
| **2048** | **2048** | **2048** | **1.01449** | **16,934.6** | **0.97242** | **17,667.1** | **0.959** |
| **4096** | **1024** | **4096** | **2.04765** | **16,780.1** | **1.84483** | **18,624.9** | **0.901** |
| **4096** | **4096** | **4096** | **7.74930** | **17,735.7** | **7.25373** | **18,947.4** | **0.936** |
| 8192 | 8192 | 8192 | 62.59013 | 17,566.9 | 57.57802 | 19,096.0 | 0.920 |

Bold rows form the predeclared large-shape corpus. Their geometric-mean ratio
is **0.940825**. The smaller and optional sustained-throughput rows are shown
for completeness and are not folded into that headline.

Throughput uses conventional GEMM algorithmic work, `2MNK/t`, with one FMA
counted as two operations. NVIDIA's
[A100 80GB datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf)
lists 19.5 TFLOP/s nominal FP32 peak for the SXM model, so the measured
17.736 TFLOP/s is 91.0% of nominal peak. The 94.08% headline is instead the
measured ratio to pedantic cuBLAS on the declared four-shape corpus.

## Kernel portfolio

For calls that require a matrix product, the frozen portfolio dispatches among
the general semantic backstop, shared-memory, register-tiled, and three SM80
asynchronous paths: wide, medium, and small. The three asynchronous kernels use
aligned vectorized access, CTA/register tiling, and two-stage 16-byte `cp.async`
pipelines. Runtime guards check shape, alignment, grid limits, compute
capability, and opt-in shared-memory support before selecting an
architecture-specific path.

The exact `512x512x512` small-kernel pocket is a useful example of why dispatch
matters: in the same canonical campaign it reached 8,147.2 GFLOP/s versus
3,643.5 GFLOP/s for the `wide` comparison variant, a **2.24x** improvement,
without broadening the route to unrelated shapes. At this size the comparison
variant dispatches its shared-memory fallback because its medium and small
asynchronous paths are disabled.

Compiler reports recorded zero spill loads and zero spill stores for every
checked kernel.

## Correctness and diagnostics

The release correctness suite passed **24/24** checked cases: 11 matrix-product
cases and 13 quick-return/scale cases. Coverage includes all transpose pairs,
padded leading dimensions, odd and tail dimensions, nontrivial `alpha` and
`beta`, beta-zero NaN poisoning, null-matrix no-ops, a non-default stream, a
skinny-tall grid-limit regression, and explicit proof of all seven required
dispatch paths:

- scale
- general
- shared
- register
- wide asynchronous
- medium asynchronous
- small asynchronous

The tested hybrid binary also passed all four NVIDIA Compute Sanitizer gates:

- `memcheck`: 0 errors
- `racecheck`: 0 hazards, 0 errors, 0 warnings
- `initcheck`: 0 errors
- `synccheck`: 0 errors

## Public evidence

The release publishes:

- deterministic source and complete evidence archives;
- all raw benchmark, build, correctness, and sanitizer logs;
- the exact tested binaries and their hashes;
- an evidence-verification report and release manifest;
- an SPDX SBOM; and
- `SHA256SUMS` covering every other release asset.

Download the assets from the
[v0.1.0 release](https://github.com/AG0708/sgBLAS/releases/tag/v0.1.0).
The verifier can re-extract the bundle, bind it to the annotated tag, regenerate
the canonical source archive, validate every artifact, and recompute the
scorecard from the raw JSONL rows.

## Limitations

- Performance claims cover one A100-SXM4-80GB, not other NVIDIA architectures.
- The headline covers four `NN` shapes, not transpose paths or arbitrary size
  distributions.
- The benchmark is same-buffer steady state, not cold-cache or end-to-end
  application latency.
- cuBLAS is intentionally restricted to pedantic FP32; this is not a comparison
  against TF32 or Tensor Core modes.
- The canonical campaign did not collect Nsight Compute hardware counters, so
  no counter claim is made. Correctness, sanitizer, compiler-resource, timing,
  telemetry, and tested-binary evidence are available.
