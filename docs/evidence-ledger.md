# sgBLAS evidence ledger

This ledger defines when a technical or quantitative sgBLAS claim is ready for
a resume, recruiter conversation, README, or public benchmark report. It is a
claim gate, not a results page. Blank fields and placeholders are intentionally
not claims.

## Rules

1. Bind every executed claim to an immutable source identity: a Git commit and
   clean status when available, plus the tuning runner's source SHA-256 before
   and after the run.
2. Report only the scope actually tested: GPU, architecture, precision, layout,
   transpose mode, shapes, `alpha`/`beta`, cache policy, and timing boundary.
3. Use `measured` for author-run numbers. Use `CI-verified` only for the checks
   that a linked CI job actually executed. Use `independently reproduced` only
   for a result rerun by an unaffiliated party.
4. A passing correctness test is not a performance result. A sanitizer pass is
   not a numerical-correctness proof. A public log is not an independent
   reproduction.
5. Do not promote numbers from `docs/results-a100-sxm4.md` into a new resume
   bullet until a fresh, complete A100 evidence bundle fills the template below.
   The curated page is useful historical context, but this ledger requires the
   raw run logs, hashes, and source binding as the promotion record.
6. Prefer `passed all checked cases` over `correct for all inputs`, and
   `reached [ratio] on [corpus]` over `matches cuBLAS`.

## Evidence labels

These labels are orthogonal. A claim can have several labels, and no label
silently implies another.

| Label | Minimum evidence | What it permits | What it does not permit |
|---|---|---|---|
| `source-supported` | The cited source revision contains the implementation, tests, and build configuration; a dependency audit supports any `from scratch` wording. | `Implemented`, `built`, or a precise architecture description. | `Passed`, `measured`, `achieved`, or any runtime number. |
| `self-measured` | A complete author-controlled run manifest, stable source hash, raw per-process logs, exact environment, successful correctness gate, and reproducible aggregation. | `Measured` or `reached`, with the exact GPU and corpus in the same sentence. | `CI-verified`, `independently verified`, universal performance, or portability to other GPUs. |
| `CI-verified` | A linked, immutable CI run at the cited commit, with the exact commands, logs, and successful exit codes retained. | Only the build/test claims executed in that job. | GPU execution or performance if CI only compiled CUDA or ran host tests; independence from the author. |
| `sanitizer-verified` | `memcheck`, `racecheck`, `initcheck`, and `synccheck` ran against the same hashed candidate binary and exited successfully; logs show zero errors, and racecheck also shows zero warnings/hazards. | `Passed all four Compute Sanitizer tools on the checked suite`. | `Memory safe`, `race free for all inputs`, numerical correctness, or performance. |
| `publicly evidenced` | Public immutable source plus a public result bundle containing the manifest, raw logs, summaries, artifact hashes, commands, and environment. Links resolve without private credentials. | `Publicly documented` or a resume link that lets a recruiter audit the claim. | `Independent` or `third-party verified` merely because the artifacts are public. |
| `externally reproduced` | An unaffiliated person or service reran the declared protocol from the cited public commit and published its own raw evidence and result. | `Independently reproduced`, scoped to that external environment and corpus. | Broader reproducibility or performance on untested architectures. |

## Claim gates

Record the evidence path or URL beside a claim before changing its status to
ready. A source line establishes implementation; an execution log establishes
what happened when that source was run.

| Potential claim | Exact gate before use | Allowed wording after the gate | Current ledger status |
|---|---|---|---|
| Built a from-scratch CUDA SGEMM library | Immutable source contains the public API, validation layer, general CUDA kernel, optimized CUDA kernels, tests, and build files. Audit the linked binary to confirm sgBLAS does not call cuBLAS or include/link CUTLASS; cuBLAS may appear only in the comparison executable. | `Built a from-scratch CUDA C++ SGEMM library` | `source-supported`; public source proof not yet recorded here. |
| Implemented a BLAS-style strict-FP32 contract | Source implements column-major `C = alpha * op(A) * op(B) + beta * C`, `NN`/`NT`/`TN`/`TT`, leading-dimension validation, quick returns, and stream-ordered launch behavior. Host/API tests pass at the cited source. Device semantics require the A100 correctness log too. | `Implemented a BLAS-style strict-FP32 SGEMM API`; add `validated` only for the executed subset. | Source paths exist; fresh execution references are blank. |
| Built a shape-dispatched SM80 kernel portfolio | Source contains explicit dispatch predicates and both optimized and semantic fallback paths. The A100 build enables the named variants for `sm_80`, completes successfully, and the binary hash is recorded. | `Built shape-dispatched SM80 kernels with correctness fallbacks` | Source-supported; fresh A100 build proof required. |
| Used a `[CTA_M]x[CTA_N]x[CTA_K]` tile, `[STAGES]`-stage `cp.async`, or `[REGISTERS]` registers/thread | Tile and stage values match the compiled definitions. Build/disassembly evidence confirms the intended kernel was emitted for `sm_80`; compiler output supplies register count and spill bytes. Dispatch evidence confirms the measured shape actually selected it. | State only the compiled values, for example `using [STAGES]-stage 16-byte cp.async`. | Source defaults are inspectable; compiled-resource and dispatch proof must come from the fresh bundle. |
| Passed correctness testing | The candidate correctness executable exits zero; its log names every case; all transpose, tail, padded-leading-dimension, nontrivial `alpha`/`beta`, quick-return, and non-default-stream cases intended for the claim are present. Record `[PASS]/[TOTAL]`. | `Passed all [TOTAL] checked device-correctness cases on [GPU]`. | Fresh result required. Never shorten to `correct for all inputs`. |
| Passed Compute Sanitizer | All four tools run on the same candidate binary/source. Each process exits zero. Logs meet the zero-error and zero-warning/hazard rule in the label table. | `Passed memcheck, racecheck, initcheck, and synccheck on the checked suite`. | Fresh result required. |
| Reached `[TFLOP/s]` at `[M]x[N]x[K]` | A complete run contains at least six balanced, separate process executions, each with at least ten warmups and the declared repeat count. CUDA-event timing excludes setup and transfer. Use the median sgBLAS GFLOP/s divided by 1,000 and report min/max or spread in the public result page. | `Measured [TFLOP/s] TFLOP/s at [shape] on one [EXACT GPU]`. | Fresh self-measured result required. |
| Reached `[PERCENT]%` of cuBLAS | The sgBLAS and cuBLAS paths use the same shape, input precision, A/B data, stream, timing method, cache treatment, and process. cuBLAS uses `CUBLAS_COMPUTE_32F_PEDANTIC`/pedantic math and `NVIDIA_TF32_OVERRIDE=0`. Aggregate the recorded per-process ratios by the declared median rule. | `Reached [PERCENT]% of strict-FP32 cuBLAS at [shape] on [GPU]`. | Fresh self-measured result required. Do not call this TF32 or Tensor Core performance. |
| Reached `[GEOMEAN]%` across the large-shape corpus | Freeze the corpus before running; retain every shape, including unfavorable results; collect the required process count for every shape; compute the geometric mean from the per-shape median ratios. Name the corpus or its shape count in the claim. | `Reached a [GEOMEAN]% geometric-mean ratio to strict-FP32 cuBLAS across [COUNT] frozen large shapes on [GPU]`. | Fresh self-measured result required. |
| Improved from `[BASELINE]` to `[CANDIDATE]` | Both variants build from the same source snapshot and environment, pass the same correctness gate, run in balanced alternating order on the same A100, and retain raw per-process values. If multiple implementation changes differ, attribute the gain to the portfolio/variant, not one technique. | `Improved the large-shape ratio from [BASELINE] to [CANDIDATE] through shape-aware dispatch and kernel tuning`. | Fresh paired A/B result required. |
| CI-verified | A public workflow at `[COMMIT]` runs the named host build/tests and, if claimed, the CUDA compile/device tests. Link the job and state exactly what ran. | `CI-verified host API tests` or `CI-verified CUDA build`; say `GPU-tested in CI` only if a GPU job executed it. | No CI evidence URL is recorded here. |
| Publicly/externally verified | For public proof, publish the complete hashed bundle and immutable source link. For external verification, additionally link the third party's independently produced bundle and compare protocols. | `Publicly documented` after publication; `independently reproduced` only after third-party execution. | Neither status is recorded here. |

### Wording that remains barred

- `cuBLAS parity`, unless a predeclared corpus-level parity interval and repeated
  measurements support it; a close point estimate at one shape is insufficient.
- `faster than cuBLAS`, unless the same fair protocol shows a statistically
  defensible advantage on every scope named in the sentence.
- `production-grade`, `drop-in BLAS replacement`, or `works on all NVIDIA GPUs`;
  the present v1 scope and evidence contract do not establish those claims.
- `verified benchmark` when the only evidence is an author-run cloud result.
- `zero bugs`, `memory safe`, or `race free`; sanitizer logs cover only the
  executed binary and inputs.

## Fresh A100 evidence record

Copy this section for each candidate. Do not delete unfilled fields; mark them
`NOT CAPTURED` so omissions remain visible.

### Identity

| Field | Value |
|---|---|
| Run ID | `[UTC_TIMESTAMP]-a100-[SOURCE_SHA256_PREFIX]` |
| Run state | `[complete / incomplete]` |
| Start / finish UTC | `[START]` / `[FINISH]` |
| Git commit | `[FULL_COMMIT_SHA or NOT AVAILABLE]` |
| Git status before / after | `[CLEAN_OR_PORCELAIN_OUTPUT]` / `[CLEAN_OR_PORCELAIN_OUTPUT]` |
| Source SHA-256 before / after | `[SOURCE_SHA256]` / `[SOURCE_SHA256]` |
| Candidate binary SHA-256 | `[BINARY_SHA256]` |
| Benchmark binary SHA-256 | `[BINARY_SHA256]` |
| Exact command | `[COMMAND]` |

Promotion gate: run state is `complete`; source hashes match; Git state did not
change; all commands are recorded with successful exit codes; binary hashes
match from first use through final artifact inventory.

### Hardware and software

| Field | Value |
|---|---|
| GPU | `[EXACT NVIDIA A100 MODEL]` |
| GPU model / compute capability / memory | `[EXACT MODEL]` / `[8.0]` / `[MEMORY]` |
| MIG / MPS / competing processes | `[STATE]` / `[STATE]` / `[STATE]` |
| Driver / CUDA toolkit / runtime / cuBLAS | `[DRIVER]` / `[TOOLKIT]` / `[RUNTIME]` / `[CUBLAS]` |
| Host OS / kernel / CPU | `[OS]` / `[KERNEL]` / `[CPU]` |
| Power limit / clocks / temperature | `[POWER]` / `[CLOCK POLICY]` / `[RANGE]` |
| CMake / compiler / build type | `[CMAKE]` / `[COMPILER]` / `[Release]` |
| CUDA architecture and definitions | `[sm_80 AND FULL -D LIST]` |
| Kernel resources | `[REGISTERS/THREAD]`, `[SHARED BYTES/CTA]`, `[SPILL STORES/LOADS]` |

### Gates

| Check | Required artifact | Result |
|---|---|---|
| Configure and build | `configure.log`, `build.log` | `[PASS/FAIL]` |
| Candidate device correctness | `hybrid/correctness.log` | `[PASS]/[TOTAL]` |
| Comparison-variant correctness | `wide/correctness.log` | `[PASS]/[TOTAL]` |
| Compiler spills | `build.log` plus manifest spill records | `[ZERO / VALUE]` |
| Compute Sanitizer memcheck | `hybrid/sanitizer-memcheck.log` | `[PASS/FAIL; ERROR COUNT; LEAK COUNT]` |
| Compute Sanitizer racecheck | `hybrid/sanitizer-racecheck.log` | `[PASS/FAIL; ERROR/WARNING/HAZARD COUNTS]` |
| Compute Sanitizer initcheck | `hybrid/sanitizer-initcheck.log` | `[PASS/FAIL; ERROR COUNT]` |
| Compute Sanitizer synccheck | `hybrid/sanitizer-synccheck.log` | `[PASS/FAIL; ERROR COUNT]` |

### Measurement protocol

| Field | Value |
|---|---|
| Math contract | `column-major NN; FP32 inputs/accumulation/output; alpha=1; beta=0` |
| cuBLAS contract | `CUBLAS_COMPUTE_32F_PEDANTIC`, pedantic math, `NVIDIA_TF32_OVERRIDE=0` |
| Cache policy | `[WARM-CACHE / OTHER, WITH EXACT PROCEDURE]` |
| Process runs | `[EVEN COUNT >= 6]` |
| Warmups / timed repeats | `[>= 10]` / `[REPEATS]` |
| Variant order | `[BALANCED ALTERNATING ORDER]` |
| Timing boundary | `[CUDA EVENTS; INCLUDED/EXCLUDED WORK]` |
| Frozen default corpus | `[LIST EVERY SHAPE]` |
| Headline large-shape corpus | `[LIST EVERY INCLUDED SHAPE]` |
| Optional sustained shape | `[SHAPE or NOT RUN]` |

### Results

| M | N | K | Runs | sgBLAS median TFLOP/s | sgBLAS min-max TFLOP/s | cuBLAS median TFLOP/s | Median ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `[M]` | `[N]` | `[K]` | `[RUNS]` | `[VALUE]` | `[MIN]-[MAX]` | `[VALUE]` | `[VALUE]` |

Large-shape geometric-mean ratio: `[VALUE]` across `[COUNT]` shapes.

Candidate-versus-comparison delta: `[VALUE]`, with variants `[CANDIDATE]` and
`[COMPARISON]`. Attribute the delta to one technique only if that technique is
the sole controlled difference.

### Provenance and visibility

| Evidence dimension | Status | Artifact or URL |
|---|---|---|
| Source-supported | `[YES/NO]` | `[IMMUTABLE SOURCE URL + PATHS]` |
| Self-measured | `[YES/NO]` | `[COMPLETE MANIFEST + RAW BUNDLE]` |
| CI-verified | `[YES/NO; EXACT SCOPE]` | `[CI RUN URL]` |
| Sanitizer-verified | `[YES/NO]` | `[FOUR LOG URLS]` |
| Publicly evidenced | `[YES/NO]` | `[PUBLIC IMMUTABLE BUNDLE URL]` |
| Externally reproduced | `[YES/NO]` | `[THIRD-PARTY SOURCE + BUNDLE URL]` |

- Claim reviewer: `[NAME]`
- Review date: `[UTC DATE]`
- Approved claims: `[CLAIM ROWS FROM THE TABLE ABOVE]`
- Known limitations: `[LIMITATIONS]`

## NVIDIA resume bullet templates

Do not replace a placeholder until its matching claim gate and evidence row are
complete.

- Built sgBLAS, a from-scratch CUDA C++ strict-FP32 SGEMM library with a BLAS-style C API and shape-dispatched SM80 kernels using register tiling, vectorized memory access, and staged `cp.async`.
- Measured `[TFLOP/S]` TFLOP/s at `[M x N x K]` on `[EXACT A100 MODEL]` (`[RATIO]%` of strict-FP32 cuBLAS; `[GEOMEAN_RATIO]%` geometric mean across `[SHAPE_COUNT]` large shapes), with `[PASSED]/[TOTAL]` device cases and `[SANITIZER_RESULT]` under Compute Sanitizer at `[SOURCE_ID]`.

If only source evidence exists, use the first bullet alone. If the fresh run is
self-measured but not public, say `measured` and omit any suggestion of external
verification. Add a repository/results link only after the public bundle gate is
complete.
