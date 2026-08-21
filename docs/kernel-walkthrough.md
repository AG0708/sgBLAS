# sgBLAS kernel and dispatch walkthrough

This document explains the CUDA implementation that is checked into this
repository. It is a source walkthrough, not a performance claim. The public
operation is column-major, strict-FP32 SGEMM:

```text
C = alpha * op(A) * op(B) + beta * C
```

Here “strict FP32” is project shorthand for FP32 operands, output,
multiplication, and accumulation with TF32 conversion disabled. It does not
mean that independent reduction orders produce bitwise-identical results.

The key implementation files are:

- `src/sgemm_validation.cpp`: classifies a call as no work, scale-only, or a
  matrix product.
- `src/sgemm_cuda.cu`: owns the public CUDA entry point, the general kernels,
  and the central product dispatcher.
- `src/sgemm_async_sm80.cu`: the 128 x 128 asynchronous-copy `NN` kernel.
- `src/sgemm_async_sm80_medium.cu`: the configurable-stage 128 x 64
  asynchronous-copy `NN` kernel.
- `src/sgemm_async_sm80_small.cu`: the 64-row asynchronous-copy `NN` kernel.

## From API call to kernel

`sgblasSgemm` first validates the operation, dimensions, leading dimensions,
and required pointers. Validation deliberately handles three non-product cases:

1. `m == 0` or `n == 0`: return success without reading scalar or matrix
   pointers.
2. `k == 0` or `alpha == 0`, with `beta == 1`: return success because `C` is
   unchanged.
3. `k == 0` or `alpha == 0`, with any other `beta`: launch `scaleKernel` and do
   not read `A` or `B`.

A product is accepted only when the handle is in `SGBLAS_MATH_FP32` mode. The
TF32 enum exists in the public API, but this implementation does not execute a
TF32 product. Work is enqueued on the handle's CUDA stream; the API does not
synchronize that stream.

The product dispatch in `launchProduct` is, in order:

```text
validated sgblasSgemm
|
+-- no work ------------------------------> return success
|
+-- scale C only -------------------------> scaleKernel
|
`-- product
    |
    +-- NT, TN, or TT --------------------> sgemmNaiveKernel
    |
    `-- NN
        |
        +-- m < 768, n < 768, or k < 32 -> sgemmSharedNnKernel
        |
        `-- otherwise
            |
            +-- guarded small SM80 pocket -> sgemmAsyncNnSmallKernel
            |
            +-- guarded medium pocket ----> sgemmAsyncNnMediumKernel
            |
            +-- guarded full wide tile ---> sgemmAsyncNnKernel
            |
            `-- fallback -----------------> sgemmRegisterNnKernel

shared/register tiled grid cannot fit either raster orientation
`------------------------------------------> sgemmNaiveKernel<NN>
```

The three asynchronous branches exist only when their CMake options compile
them into the library. Plain CMake defaults all three options to `OFF`; the
checked-in `cuda-release` preset turns all three on.

## Kernel portfolio at the checked-in preset defaults

The table distinguishes logical output tiles from K tiles. "Accumulators" are
the per-thread elements of `C` held across the K loop.

| Kernel family | CTA output tile and K tile | CTA threads | Per-thread output mapping | Shared memory per CTA | Edge handling |
|---|---:|---:|---:|---:|---|
| `scaleKernel` | no GEMM tile | 16 x 16 | one column and grid-strided rows | none | bounds check |
| `sgemmNaiveKernel` | one output at a time | 16 x 16 | one accumulator per visited output | none | bounds check; all transpose pairs |
| `sgemmSharedNnKernel` | 32 x 32 x 32 | 32 x 8 = 256 | 1 x 4 = 4 accumulators | 8,448 B static | zero-filled loads and predicated stores |
| `sgemmRegisterNnKernel` | 128 x 128 x 32 | 256 | 8 x 8 = 64 accumulators | 33,408 B static | scalar tail loads and predicated stores |
| `sgemmAsyncNnSmallKernel` | 64 x 32 x 32 | 128 | 2 x 8 = 16 accumulators | 24,576 B dynamic | full tiles only |
| `sgemmAsyncNnMediumKernel` | 128 x 64 x 32 | 256 | 4 x 8 = 32 accumulators | 49,152 B dynamic | full tiles only |
| `sgemmAsyncNnKernel` | 128 x 128 x 32 | 256 | 8 x 8 = 64 accumulators | 65,536 B dynamic | full tiles only |

The small and medium rows use the `cuda-release` values: 32 small-tile columns,
32 logical thread rows, and two medium pipeline stages. Their CMake knobs can
change the thread mapping or shared-memory footprint. The byte counts come
directly from the declared arrays:

```text
shared NN:    (32 * 33 + 32 * 33) floats                  =  8,448 B
register NN:  (32 * 129 + 128 * 33) floats                = 33,408 B
small async:  2 * (64 * 32 + 32 * 32) floats             = 24,576 B
medium async: 2 * (128 * 32 + 64 * 32) floats            = 49,152 B
wide async:   2 * (128 * 32 + 128 * 32) floats           = 65,536 B
```

`__launch_bounds__` encodes compiler constraints, not measured occupancy. The
register, medium, wide, and small kernels request minimum blocks per SM of 2,
2, 1, and a configurable value respectively; the preset supplies 5 for the
small kernel.

## The semantic backstops

### Scale-only kernel

`scaleKernel` uses a 16 x 16 block. `threadIdx.x` selects a column and
`threadIdx.y` selects the first row. Grid Y is capped at 65,535, so a thread
grid-strides over additional rows when necessary. The `beta == 0` template
writes zero without reading `C`; the general template multiplies the old `C` by
`beta`.

### Naive SGEMM

`sgemmNaiveKernel` is templated on `TransposeA`, `TransposeB`, and `BetaMode`.
One thread computes one output at a time with a serial K loop, then grid-strides
over more rows if grid Y was capped. Its source indices encode the column-major
contract directly:

```text
A, N: A[row + reduction * lda]
A, T: A[reduction + row * lda]
B, N: B[reduction + column * ldb]
B, T: B[column + reduction * ldb]
C:    C[row + column * ldc]
```

This is the path for `NT`, `TN`, and `TT`. It is also the final `NN` fallback if
a 2-D tiled grid cannot place either tile count in CUDA grid Y.

Three compile-time beta variants keep the epilogue exact:

- zero: `C = alpha * accumulator`, with no read of `C`;
- one: `C += alpha * accumulator`;
- general: `C = fma(beta, C, alpha * accumulator)`.

Every product kernel uses the same three-way beta specialization.

## Synchronous tiled `NN` kernels

### 32 x 32 shared-memory kernel

A 32 x 8 block computes one 32 x 32 output tile over K in chunks of 32.
For `(threadIdx.x, threadIdx.y) = (r, q)`:

```text
output row:     tile_row + r
output columns: tile_column + q + {0, 8, 16, 24}
```

Each thread owns four accumulators. On each K tile, its linear thread index
participates in four iterations of the cooperative-load loop. Each iteration
loads one `A` scalar and one `B` scalar, so the CTA loads each 32 x 32 input
tile once. Out-of-range elements are written as zero into shared memory.

The arrays are declared as `A[32][33]` and `B[32][33]`. The extra element in
the inner dimension is padding. After a CTA barrier, each of 32 reduction steps
broadcasts one `A` value to the thread's four `B` values and performs four
FP32 FMAs. Stores are predicated for M and N tails.

### 128 x 128 register-blocked kernel

A 256-thread block is treated as a logical 16 x 16 thread grid:

```text
thread_row    = thread_index % 16
thread_column = thread_index / 16
```

Thread `(r, q)` owns the Cartesian product of these rows and columns:

```text
rows:    tile_row    + r + {0, 16, 32, 48, 64, 80, 96, 112}
columns: tile_column + q + {0, 16, 32, 48, 64, 80, 96, 112}
```

That is an 8 x 8 register tile, or 64 accumulators per thread. For each K tile:

1. Every thread executes four cooperative-load iterations.
2. Each iteration obtains one `float4` from `A` and one `float4` from `B`, for
   16 `A` and 16 `B` values per thread across the tile load.
3. A 16-byte-aligned source uses a vector load. An unaligned source is assembled
   from four scalar loads. M, N, and K tails are zero-filled.
4. After the barrier, each reduction step reads eight `A` and eight `B` values
   from shared memory and computes their 8 x 8 outer product: 64 FP32 FMAs.
5. A second barrier protects the shared arrays before the next K tile replaces
   them.

The dispatcher selects this kernel for `NN` when `m >= 768`, `n >= 768`, and
`k >= 32`, unless an enabled asynchronous kernel wins first. Unlike the async
kernels, it accepts partial M, N, and K tiles.

## Asynchronous-copy `NN` kernels

All three asynchronous families use inline PTX `cp.async.cg.shared.global` to
copy 16 bytes from global memory into shared memory. A group is committed after
the cooperative tile load. The kernel waits for the required group and then
uses `__syncthreads()` before consuming that stage.

Their shared-memory stages contain dense, unpadded `A` and `B` tiles. They do
not predicate loads or stores, which is why dispatch proves all of the
following before entering any async family:

- the operation is `NN` and passed the outer `m >= 768`, `n >= 768`, `k >= 32`
  gate;
- M, N, and K are exact multiples of that family's tile dimensions;
- `A` and `B` base addresses are 16-byte aligned;
- `lda` and `ldb` are multiples of four floats;
- at least one grid raster orientation fits the 65,535 grid-Y limit;
- the active device has compute-capability major version at least 8;
- the device's opt-in shared-memory-per-block limit covers the family footprint.

The selector does not require aligned `C`: epilogue stores are scalar.
Although the files and symbols say `sm80`, the runtime test is `major >= 8`, not
an equality check for SM80.

### Wide: 128 x 128, two stages

The wide kernel has the same logical 16 x 16 thread grid and 8 x 8 accumulator
mapping as the synchronous register kernel. Each stage holds a 128 x 32 `A`
tile and a 32 x 128 `B` tile. Each thread issues four 16-byte copies for `A`
and four for `B`.

Stage 0 is loaded and made visible before the loop. While a thread computes the
current K tile, it has already issued copies for the next tile into `stage ^ 1`.
At the tile boundary, the CTA waits for that copy group, synchronizes, flips the
stage bit, and continues. The 64 KiB allocation is two complete A/B stages.

### Medium: 128 x 64, two or three stages

At preset defaults, the medium kernel maps 256 threads as 32 logical thread
rows by 8 logical thread columns. Each thread therefore owns four output rows
spaced by 32 and eight output columns spaced by 8, for 32 accumulators.

One K stage contains 4,096 `A` floats and 2,048 `B` floats. At defaults, each
thread issues four 16-byte A copies and two 16-byte B copies. The two-stage
pipeline alternates stages exactly like the wide kernel and uses 48 KiB.

`SGBLAS_SM80_MEDIUM_STAGES=3` instead allocates 72 KiB, primes stages 0 and 1,
and uses `cp.async.wait_group 1` while a third stage is in flight. The source
accepts only two or three stages. A separate compile-time knob selects no L2
prefetch hint, `.L2::128B`, or `.L2::256B`; the preset selects no hint.

### Small: 64 x 32, two stages at defaults

At preset defaults, 128 threads form 32 logical thread rows by 4 logical thread
columns. Each thread owns two rows spaced by 32 and eight columns spaced by 4,
for 16 accumulators. Each stage contains 2,048 `A` floats and 1,024 `B` floats;
each thread issues four 16-byte A copies and two 16-byte B copies. The two-stage
allocation is 24 KiB.

The tile-column count, logical thread-row count, and launch-bounds minimum are
compile-time knobs. The preset values are 32, 32, and 5 respectively. The
small kernel changes the number and shape of CTAs; the launch-bounds value is a
compiler constraint and does not prove five resident blocks at runtime.

## Exact async dispatch order and default pockets

The common quantities are:

```text
full wide tile = (m % 128 == 0) && (n % 128 == 0) && (k % 32 == 0)
aligned        = 16-byte-aligned A and B && lda % 4 == 0 && ldb % 4 == 0
wide_ctas      = (m / 128) * (n / 128) using integer division
```

When all three async options are enabled, the first matching branch wins:

1. **Small.** At defaults: `m % 64 == 0`, `n % 32 == 0`, `k % 32 == 0`,
   `k >= 128`, and either `wide_ctas <= 128`, or all of
   `196 <= wide_ctas <= 256`, `m <= 2048`, and `n <= 4096`.
2. **Medium.** `m % 128 == 0`, `n % 64 == 0`, `k % 32 == 0`, `k >= 64`, and
   `128 <= wide_ctas <= 2147483647` at defaults.
3. **Wide.** M and N are multiples of 128 and K is a multiple of 32.
4. **Register fallback.** Any otherwise-valid large `NN` shape, including
   unaligned or partial tiles.

Every async branch additionally applies the common alignment, grid-fit, device,
and shared-memory checks above. The small and medium thresholds are CMake cache
variables, so the numbers in this section describe the checked-in defaults,
not immutable ABI behavior.

The tiled launchers normally place row tiles in grid X and column tiles in grid
Y. They flip the axes if column tiles would exceed 65,535 and row tiles fit.
The medium kernel can prefer the opposite raster through
`SGBLAS_SM80_MEDIUM_N_MAJOR_RASTER`, while still falling back to the other
orientation when the preferred grid Y would be too large. If both tile counts
exceed 65,535, the async path is rejected. The shared and register launchers
fall back to the naive `NN` kernel in the equivalent case.

## Interview-ready trace: aligned 4096 cubed `NN`

Assume the checked-in `cuda-release` preset, `m = n = k = 4096`,
`lda = ldb = ldc = 4096`, 16-byte-aligned `A` and `B`, nonzero `alpha`,
`beta = 0`, and an eligible device with compute-capability major version at
least 8 and at least 48 KiB of opt-in shared memory.

1. Validation classifies the call as a product. FP32 mode passes.
2. `NN` and all three dimensions pass the 768/768/32 outer gate.
3. `wide_ctas = (4096 / 128) * (4096 / 128) = 1,024`.
4. The small branch fails both default CTA pockets: 1,024 is neither at most
   128 nor in 196 through 256.
5. The medium branch passes: the dimensions are multiples of 128, 64, and 32;
   K is at least 64; 1,024 is above its default minimum of 128; alignment,
   grid, device, and 48 KiB shared-memory checks pass.
6. Medium appears before wide in dispatch, so the selected launch is a
   32 x 64 grid of 256-thread CTAs. There are 2,048 CTAs, each producing a
   128 x 64 tile.
7. For one CTA, thread 37 has `thread_row = 37 % 32 = 5` and
   `thread_column = 37 / 32 = 1`. Relative to the CTA origin it owns rows
   `{5, 37, 69, 101}` and columns
   `{1, 9, 17, 25, 33, 41, 49, 57}`: 32 outputs.
8. For each 32-wide K tile, that thread issues four 16-byte A copies and two
   16-byte B copies for the cooperative next-stage load. At each of 32
   reduction positions, it reads four A values and eight B values and computes
   their outer product, or 32 FMAs. That is 1,024 FMAs per thread per K tile.
9. K has 128 such tiles. After the primed first tile, copies for the next tile
   are issued into the alternate shared stage before the current tile's FMAs.
10. The `beta == 0` specialization writes `alpha * accumulator` and never reads
    the old `C`. The API returns the launch status; completion and any later
    execution fault are observed when the caller subsequently synchronizes the
    stream.

For contrast, changing the operation to `NT` bypasses every tiled `NN` kernel
and instantiates `sgemmNaiveKernel<false, true, BetaMode>`. That sharp boundary
is intentional in the current source: transpose correctness is general, while
the optimized portfolio is deliberately narrow.

## What this implementation does and does not establish

The source establishes a strict-FP32, stream-ordered, column-major SGEMM path
with all four transpose combinations, arbitrary legal leading dimensions, and
tail-safe synchronous fallbacks. It also establishes guarded, full-tile async
specializations for aligned `NN` inputs on devices passing the runtime checks.

The source alone does **not** establish achieved occupancy, cache hit rates,
instruction issue efficiency, portability of one tuning pocket to another GPU,
or any percentage of cuBLAS performance. Those are measurement questions. It
also does not implement a TF32 product, Tensor Core kernels, row-major public
storage, batched GEMM, or optimized transpose kernels.
