# RunPod target

Use RunPod for every CUDA correctness or performance decision. The local Apple
Silicon machine is only a development and host-test environment.

## First target

Select the exact RunPod GPU type `NVIDIA A100-SXM4-80GB` (`sm_80`) and verify
that exact `nvidia-smi` name before entering a result in the SXM4 scorecard.
RunPod also offers A100 PCIe machines; keep those results separate.

Use at least 20 GB of volume disk mounted at `/workspace`. Volume disk survives
a stopped Pod and continues billing, but is deleted when the Pod is terminated;
copy evidence off the Pod before termination. The repository expects a CUDA
12.8 development environment; `Dockerfile` records the version-tagged base image
and host tools used for current-code validation.

The archived A100 snapshot used CUDA compiler 12.4.131, cuBLAS 12.4.5, and
Ubuntu 22.04. The current CUDA 12.8.1/Ubuntu 24.04 image is therefore a new
benchmark environment, not an exact reproduction of that snapshot; publish a
new scorecard for it.

## Bootstrap

Once the repository is present at `/workspace/sgblas`:

```bash
cd /workspace/sgblas
mkdir -p results
./tools/probe_target.sh | tee results/target.txt
cmake --preset cuda-release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build --preset cuda-release -j
ctest --preset cuda-release --output-on-failure
```

The first GPU run is a correctness and benchmark-baseline run, not a tuning run.
Save the target probe, test output, benchmark CSV, CUDA version, and Git commit
together. Only after that baseline is clean should profiler-driven kernel changes
begin.

## Moving an immutable source revision

Clone the public repository and check out the exact release candidate commit:

```bash
git clone https://github.com/AG0708/sgBLAS.git /workspace/sgblas
cd /workspace/sgblas
git checkout --detach RELEASE_CANDIDATE_SHA
test -z "$(git status --porcelain)"
```

The runner rejects a missing commit, a dirty checkout, and source changes during
the campaign.

## Colab fallback

Colab is useful for a quick compile-and-correctness smoke test, but its GPU model,
clocks, runtime lifetime, and background load can change. Do not use Colab numbers
for headline performance claims or autotuning tables.
