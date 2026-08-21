#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
image_name=${SGBLAS_CUDA_IMAGE_NAME:-sgblas-dev:cuda12.8}
cuda_architecture=${SGBLAS_CUDA_ARCHITECTURE:-80}
experimental_sm80_async=${SGBLAS_EXPERIMENTAL_SM80_ASYNC:-OFF}
experimental_sm80_medium=${SGBLAS_EXPERIMENTAL_SM80_MEDIUM:-OFF}
experimental_sm80_small=${SGBLAS_EXPERIMENTAL_SM80_SMALL:-OFF}
medium_thread_rows=${SGBLAS_SM80_MEDIUM_THREAD_ROWS:-32}
medium_n_major_raster=${SGBLAS_SM80_MEDIUM_N_MAJOR_RASTER:-OFF}
medium_l2_prefetch_bytes=${SGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES:-0}
medium_stages=${SGBLAS_SM80_MEDIUM_STAGES:-2}
small_tile_columns=${SGBLAS_SM80_SMALL_TILE_COLUMNS:-32}
small_thread_rows=${SGBLAS_SM80_SMALL_THREAD_ROWS:-32}
small_max_wide_ctas=${SGBLAS_SM80_SMALL_MAX_WIDE_CTAS:-128}
small_second_min_wide_ctas=${SGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS:-196}
small_second_max_wide_ctas=${SGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS:-256}
small_second_max_m=${SGBLAS_SM80_SMALL_SECOND_MAX_M:-2048}
small_second_max_n=${SGBLAS_SM80_SMALL_SECOND_MAX_N:-4096}

# Codex and other non-login shells may find the Docker CLI but not Docker
# Desktop's configured credential helper.
docker_desktop_bin=/Applications/Docker.app/Contents/Resources/bin
if [[ -d "${docker_desktop_bin}" ]]; then
  export PATH="${docker_desktop_bin}:${PATH}"
fi

docker build \
  --file "${repo_root}/infra/runpod/Dockerfile" \
  --tag "${image_name}" \
  "${repo_root}"

docker run --rm \
  --volume "${repo_root}:/workspace/sgblas" \
  --workdir /workspace/sgblas \
  "${image_name}" \
  bash -lc \
    "cmake --preset cuda-release -DCMAKE_CUDA_ARCHITECTURES=${cuda_architecture} -DSGBLAS_EXPERIMENTAL_SM80_ASYNC=${experimental_sm80_async} -DSGBLAS_EXPERIMENTAL_SM80_MEDIUM=${experimental_sm80_medium} -DSGBLAS_EXPERIMENTAL_SM80_SMALL=${experimental_sm80_small} -DSGBLAS_SM80_MEDIUM_THREAD_ROWS=${medium_thread_rows} -DSGBLAS_SM80_MEDIUM_N_MAJOR_RASTER=${medium_n_major_raster} -DSGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES=${medium_l2_prefetch_bytes} -DSGBLAS_SM80_MEDIUM_STAGES=${medium_stages} -DSGBLAS_SM80_SMALL_TILE_COLUMNS=${small_tile_columns} -DSGBLAS_SM80_SMALL_THREAD_ROWS=${small_thread_rows} -DSGBLAS_SM80_SMALL_MAX_WIDE_CTAS=${small_max_wide_ctas} -DSGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS=${small_second_min_wide_ctas} -DSGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS=${small_second_max_wide_ctas} -DSGBLAS_SM80_SMALL_SECOND_MAX_M=${small_second_max_m} -DSGBLAS_SM80_SMALL_SECOND_MAX_N=${small_second_max_n} && cmake --build --preset cuda-release --target sgblas_cuda_correctness sgblas_benchmark -j2"
