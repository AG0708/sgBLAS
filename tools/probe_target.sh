#!/usr/bin/env bash
set -euo pipefail

for command_name in nvidia-smi nvcc cmake; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "missing_required_command=${command_name}" >&2
    exit 1
  fi
done

echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit=$(git rev-parse --verify HEAD 2>/dev/null || echo uncommitted)"
echo "kernel=$(uname -srmo)"

nvcc --version | sed 's/^/nvcc=/'

echo "gpu_fields=index,name,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory"
nvidia-smi \
  --query-gpu=index,name,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory \
  --format=csv,noheader | sed 's/^/gpu=/'

if command -v ncu >/dev/null 2>&1; then
  ncu --version | sed 's/^/ncu=/'
else
  echo "ncu=not-found"
fi

if command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --version | sed 's/^/compute_sanitizer=/'
else
  echo "compute_sanitizer=not-found"
fi
