# Design contract

## Goal and boundary

The v1 deliverable is a correct, stream-aware, column-major SGEMM implementation with a fast `NN` specialization. It is intentionally not a complete BLAS library and not a portable claim of cuBLAS parity.

The design separates three concerns:

1. a stable C API and BLAS-compatible validation rules;
2. a general correctness kernel that covers the full v1 contract;
3. guarded, GPU-specific optimized kernels selected only when their preconditions hold.

CUTLASS is used as a readable reference for hierarchical decomposition, pipelines, epilogues, and schedulers, but sgBLAS does not link to or include CUTLASS. NVIDIA documents the same threadblock/warp/thread hierarchy in [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).

## v1 SGEMM contract

The operation is

```text
C(m,n) = alpha * op(A)(m,k) * op(B)(k,n) + beta * C(m,n)
```

with these rules:

- `A`, `B`, and `C` contain IEEE binary32 values.
- The selected v1 math mode performs FP32 multiplication and FP32 accumulation. Fused FP32 FMA is permitted; reduced-precision TF32 inputs are not. “Strict FP32” is project shorthand for this precision contract and does not imply bitwise-identical results or an identical reduction order.
- Matrices use column-major storage.
- `transa` and `transb` independently accept `N` or `T`, covering `NN`, `NT`, `TN`, and `TT`.
- `m`, `n`, and `k` may be any non-negative `int` values.
- `lda >= max(1, transa == N ? m : k)`.
- `ldb >= max(1, transb == N ? k : n)`.
- `ldc >= max(1, m)`.
- `alpha` and `beta` are valid host pointers for the duration of the call.
- Matrix pointers identify device memory accessible from the handle's current device. A, B, and C storage must not overlap.
- The operation is enqueued on the handle's stream and does not introduce a device- or stream-wide synchronization.
- If `m == 0` or `n == 0`, the call succeeds without launching work.
- If `k == 0` or `alpha == 0`, A and B are not read and the result is `beta * C`.
- If `beta == 0`, the prior values, including NaNs, in C are not read.

The public definitions live in `include/sgblas/sgblas.h`. Invalid enums, dimensions, leading dimensions, handles, or required pointers return a status code rather than terminating the process.

## Kernel and dispatch structure

The general kernel is the semantic backstop. It handles all transpose modes, legal tails, and legal leading dimensions. Optimized kernels may make stronger assumptions, but those assumptions belong in an explicit dispatch predicate; they must never leak into the public contract.

The initial dispatch order is:

```text
validated request
    -> optimized FP32 NN kernel when its GPU/layout/alignment/shape guard passes
    -> general FP32 kernel otherwise
```

Future kernels should be keyed by at least compute capability, math mode, transpose pair, alignment class, and coarse M/N/K shape bucket. Tile dimensions, stage count, or vector width are implementation details, not ABI.

## Tuning ladder

Each rung must preserve correctness and demonstrate a benchmark improvement before becoming the new baseline.

### Portable FP32 foundation

1. Map adjacent lanes to adjacent elements of C and coalesce A/B/C global accesses.
2. Tile M, N, and K at CTA scope; load each A/B tile once into shared memory.
3. Pad or swizzle shared layouts to avoid bank conflicts. NVIDIA explains coalescing, redundant-load elimination, shared-memory banks, and padding in the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
4. Use naturally aligned 8- or 16-byte memory operations where the dispatch guard proves alignment. Keep a predicated/scalar tail. NVIDIA's [vectorized memory-access guidance](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) emphasizes both the instruction-count benefit and alignment requirement.
5. Give each thread a 2D register accumulator tile and reuse loaded A/B values across an outer product. Unroll the inner K fragment.
6. Double-buffer shared tiles and register fragments so data movement overlaps useful FMA work.
7. Make the epilogue coalesce stores and handle `alpha`/`beta` without a second global-memory pass.
8. Tune CTA, warp, and thread tiles together. Do not maximize occupancy blindly: NVIDIA notes that extra registers and instruction-level parallelism can outperform higher occupancy in the [occupancy guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

### GPU-specific progression

| Target | Strict-FP32 path | Architecture-specific next step |
|---|---|---|
| Volta/Turing (`sm_70`-`sm_75`) | Shared-memory multibuffering and FP32 CUDA-core FMA | Tune synchronous global-to-shared staging and register reuse. CUDA 13 removed offline compilation and library support for Volta, so `sm_70` requires a CUDA 12.x toolchain. |
| Ampere/Ada (compute capability 8.x; `sm_80`-`sm_89`) | FP32 CUDA-core FMA remains the scored v1 path | On `sm_80` and later, use aligned 16-byte PTX `cp.async`/LDGSTS or CUDA pipeline APIs to overlap global-to-shared copies. Published tuning is A100 compute capability 8.0 only. See [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html). |
| Hopper (`sm90a`) | Preserve a true-FP32 CUDA-core lane | Explore TMA, warp specialization, and persistent scheduling; WGMMA belongs to a separately scored Tensor Core mode. See the [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) |
| Blackwell datacenter (`sm100a`/`sm103a`) | Preserve a true-FP32 lane | Explore TMA, TMEM, CLC scheduling, and `tcgen05` only in a declared Tensor Core or emulation mode. See NVIDIA's [tcgen05 guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html) |
| Blackwell GeForce (`sm120`) | Preserve a true-FP32 lane | Treat it as a separate backend: SM120 uses extended `mma.sync` for narrow types and lacks SM100-style multicast. See [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) |

Architecture specialization must be compiled for the matching feature target. An `sm90a` or `sm100a` kernel is not a generic fallback for all devices carrying the same marketing family name.

## FP32 and TF32 are different products

`SGBLAS_MATH_FP32` means FP32 inputs reach FP32 arithmetic without TF32 truncation. A future `SGBLAS_MATH_TF32` mode may use Tensor Cores, but it must have its own dispatch table, error tolerance, cuBLAS compute mode, and scorecard.

The benchmark must never divide an FP32 kernel's throughput by a TF32 cuBLAS result, or vice versa. NVIDIA's [cuBLAS compute-type documentation](https://docs.nvidia.com/cuda/cublas/) distinguishes `CUBLAS_COMPUTE_32F_PEDANTIC`, `CUBLAS_COMPUTE_32F`, and `CUBLAS_COMPUTE_32F_FAST_TF32`, and notes that `NVIDIA_TF32_OVERRIDE=0` disables TF32 acceleration across NVIDIA libraries.

## Correctness strategy

- Use a higher-precision host reference for small matrices and strict-FP32 cuBLAS as a large-case cross-check.
- Compare with a magnitude- and K-aware tolerance; floating-point reductions are not expected to be bitwise identical when their legal evaluation orders differ.
- Include zero dimensions, K tails, non-tile M/N tails, padded leading dimensions, every transpose pair, `alpha` in `{0, 1, nontrivial}`, and `beta` in `{0, 1, nontrivial}`.
- Seed random cases and print the seed on failure.
- Test a non-default, nonblocking CUDA stream and verify that the API does not synchronize unrelated work.
- Run NVIDIA Compute Sanitizer during development in addition to numerical tests.

NVIDIA recommends reference comparison after every optimization and explains why epsilon-based comparison is often required in [Getting the Right Answer](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

## Explicit non-goals for v1

- batched, grouped, distributed, sparse, complex, FP64, FP16, BF16, FP8, or FP4 GEMM;
- row-major public storage;
- fused bias or activation epilogues;
- universal parity with cuBLAS on every size;
- Tensor Core performance reported as strict FP32;
- hidden JIT compilation or runtime autotuning in a timed region.
