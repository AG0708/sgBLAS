from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_evidence", REPOSITORY / "tools" / "verify_evidence.py"
)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": verify.sha256_file(path),
    }


def artifact_file_record(path: Path, run_root: Path) -> dict[str, object]:
    record = file_record(path)
    record["path"] = path.resolve().relative_to(run_root.resolve()).as_posix()
    return record


def copy_proof(original: Path, copied: Path, run_root: Path) -> dict[str, object]:
    original_record = file_record(original)
    copied_record = artifact_file_record(copied, run_root)
    return {
        "original": original_record,
        "copied": copied_record,
        "sha256_equal": original_record["sha256"] == copied_record["sha256"],
        "size_equal": original_record["size_bytes"] == copied_record["size_bytes"],
    }


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        runs: int = 6,
        balanced: bool = True,
        missing_shape: bool = False,
        bad_sanitizer: str | None = None,
        bad_correctness: bool = False,
    ) -> None:
        self.root = root
        self.source = root / "source"
        self.build = root / "build" / "campaign"
        self.run = root / "artifacts" / "run"
        self.runs = runs
        self.command_counter = 0
        self.commands: list[dict[str, object]] = []
        self.values: dict[
            str, list[dict[tuple[int, int, int], dict[str, float]]]
        ] = {variant: [] for variant in verify.VARIANTS}

        self._make_source()
        self._make_binaries()
        self.full_order = [
            ["wide", "hybrid"] if index % 2 == 0 else ["hybrid", "wide"]
            for index in range(runs)
        ]
        self.timing_full = {
            variant: [
                "sgblas-first"
                if (index + offset) % 2 == 0
                else "cublas-first"
                for index in range(runs)
            ]
            for variant, offset in (("wide", 0), ("hybrid", 1))
        }
        self.prime_order = {"wide": "sgblas-first", "hybrid": "cublas-first"}
        self._make_build_logs()
        self._make_correctness(bad_correctness)
        self._make_prime()
        self._make_benchmarks(balanced, missing_shape)
        self._make_sanitizers(bad_sanitizer)
        self._make_summary()
        self.manifest = self._make_manifest()
        self._finalize_inventory()

    def _make_source(self) -> None:
        write(self.source / ".dockerignore", "build\nresults\n")
        write(self.source / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.24)\n")
        write(self.source / "CMakePresets.json", "{}\n")
        write(self.source / "LICENSE", "synthetic license\n")
        write(self.source / "PROVENANCE.md", "# Synthetic provenance\n")
        write(self.source / "THIRD_PARTY_NOTICES.md", "# Synthetic notices\n")
        write(self.source / "bench" / "benchmark.cu", "// synthetic benchmark\n")
        write(self.source / "cmake" / "config.cmake", "# synthetic config\n")
        write(self.source / "include" / "sgblas.h", "// synthetic header\n")
        write(self.source / "infra" / "runpod" / "Dockerfile", "FROM scratch\n")
        write(self.source / "src" / "kernel.cu", "// synthetic source\n")
        write(self.source / "tests" / "correctness.cpp", "// synthetic test\n")
        write(self.source / "tools" / "verify_evidence.py", "# synthetic verifier\n")
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        subprocess.run(
            ["git", "-C", str(self.source), "config", "user.email", "fixture@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.source), "config", "user.name", "Fixture"],
            check=True,
        )
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.source), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        self.commit = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        self.source_digest, self.source_files = verify.source_snapshot(self.source)

    def _make_binaries(self) -> None:
        self.binaries: dict[str, dict[str, Path]] = {}
        self.copied_binaries: dict[str, dict[str, Path]] = {}
        self.cmake_caches: dict[str, tuple[Path, Path]] = {}
        for variant in verify.VARIANTS:
            self.binaries[variant] = {}
            self.copied_binaries[variant] = {}
            for kind in ("benchmark", "correctness"):
                filename = (
                    "sgblas_cuda_correctness" if kind == "correctness" else "sgblas_benchmark"
                )
                path = self.build / variant / filename
                write(path, f"synthetic {variant} {kind} binary\n")
                path.chmod(0o755)
                self.binaries[variant][kind] = path
                copied = self.run / variant / "tested-binaries" / filename
                write(copied, path.read_text(encoding="utf-8"))
                copied.chmod(0o755)
                self.copied_binaries[variant][kind] = copied
            cache_lines = [
                "CMAKE_BUILD_TYPE:STRING=Release",
                "CMAKE_CUDA_ARCHITECTURES:STRING=80",
                "SGBLAS_ENABLE_CUDA:BOOL=ON",
                "SGBLAS_BUILD_TESTS:BOOL=ON",
                "SGBLAS_BUILD_BENCHMARKS:BOOL=ON",
            ]
            cache_lines.extend(
                f"{definition[2:].split('=', 1)[0]}:STRING={definition.split('=', 1)[1]}"
                for definition in verify.EXPECTED_VARIANTS[variant]
            )
            cache_text = "\n".join(cache_lines) + "\n"
            original_cache = self.build / variant / "CMakeCache.txt"
            copied_cache = self.run / variant / "tested-binaries" / "CMakeCache.txt"
            write(original_cache, cache_text)
            write(copied_cache, cache_text)
            self.cmake_caches[variant] = (original_cache, copied_cache)

    def _make_build_logs(self) -> None:
        for variant in verify.VARIANTS:
            definitions = verify.EXPECTED_VARIANTS[variant]
            configure = [
                "/usr/local/bin/cmake",
                "-S",
                str(self.source),
                "-B",
                str(self.build / variant),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_ARCHITECTURES=80",
                "-DSGBLAS_ENABLE_CUDA=ON",
                "-DSGBLAS_BUILD_TESTS=ON",
                "-DSGBLAS_BUILD_BENCHMARKS=ON",
                *definitions,
            ]
            self._logged(
                self.run / variant / "configure.log",
                configure,
                "-- Configuring done\n",
            )
            self._logged(
                self.run / variant / "build.log",
                [
                    "/usr/local/bin/cmake",
                    "--build",
                    str(self.build / variant),
                    "-j",
                    "2",
                ],
                "ptxas info    : 0 bytes spill stores, 0 bytes spill loads\n",
            )

    def _timestamp(self, offset: int = 0) -> str:
        moment = dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=self.command_counter * 3 + offset
        )
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _logged(
        self, path: Path, argv: list[str], body: str, *, environment: str = "base"
    ) -> None:
        self.command_counter += 1
        started = self._timestamp()
        finished = self._timestamp(1)
        contents = (
            "# command: "
            + json.dumps(argv)
            + f"\n# started_utc: {started}\n"
            + body.rstrip("\n")
            + f"\n# finished_utc: {finished}\n# exit_code: 0\n"
        )
        write(path, contents)
        self.commands.append(
            {
                "argv": argv,
                "cwd": str(self.source),
                "environment": environment,
                "started_utc": started,
                "finished_utc": finished,
                "duration_seconds": 1.0,
                "state": "complete",
                "exit_code": 0,
                "log_path": path.resolve().relative_to(self.run.resolve()).as_posix(),
                "log": artifact_file_record(path, self.run),
            }
        )

    @staticmethod
    def _correctness_text(success: bool = True) -> str:
        ending = (
            "All 10 matrix-product cases and 13 quick-return/scale cases passed."
            if success
            else "One or more sgBLAS SGEMM correctness cases failed."
        )
        return "NN odd beta=0 PASS\nNN medium async tile PASS\n" + ending + "\n"

    def _make_correctness(self, bad_correctness: bool) -> None:
        for variant in verify.VARIANTS:
            self._logged(
                self.run / variant / "correctness.log",
                [str(self.binaries[variant]["correctness"])],
                self._correctness_text(not (bad_correctness and variant == "hybrid")),
            )

    @staticmethod
    def _metadata(argv: list[str], order: str, warmups: int, repeats: int, seed: int) -> dict[str, object]:
        return {
            "record_type": "metadata",
            "schema_version": 1,
            "benchmark": {
                "timed_order": order,
                "warmup_order": "alternating-per-launch",
                "warmups_per_implementation": warmups,
                "timed_repeats_per_implementation": repeats,
                "seed": seed,
                "cache_policy": "same-buffer-steady-state",
                "stream": "shared-nonblocking",
                "sgblas_math_mode": "SGBLAS_MATH_FP32",
                "cublas_math_mode": "CUBLAS_PEDANTIC_MATH",
                "cublas_compute_type": "CUBLAS_COMPUTE_32F_PEDANTIC",
                "cublas_algorithm": "CUBLAS_GEMM_DEFAULT",
                "nvidia_tf32_override": "0",
            },
            "cuda": {
                "driver_version_raw": 12080,
                "driver_version": "12.8",
                "runtime_version_raw": 12080,
                "runtime_version": "12.8",
            },
            "cublas": {"version_raw": 120805, "version": "12.8.5"},
            "device": {
                "ordinal": 0,
                "name": "NVIDIA A100-SXM4-80GB",
                "pci_bus_id": "0000:00:00.0",
                "compute_capability_major": 8,
                "compute_capability_minor": 0,
                "multiprocessor_count": 108,
                "total_global_memory_bytes": 85_000_000_000,
                "l2_cache_bytes": 40_000_000,
                "shared_memory_per_block_bytes": 49_152,
                "shared_memory_per_block_optin_bytes": 163_840,
                "shared_memory_per_multiprocessor_bytes": 167_936,
                "registers_per_multiprocessor": 65_536,
                "warp_size": 32,
                "max_threads_per_multiprocessor": 2_048,
                "reported_core_clock_khz": 1_410_000,
                "reported_memory_clock_khz": 1_593_000,
                "memory_bus_width_bits": 5_120,
            },
            "argv": argv,
        }

    @staticmethod
    def _result(
        shape: tuple[int, int, int], order: str, warmups: int, repeats: int, factor: float
    ) -> tuple[dict[str, object], dict[str, float]]:
        m, n, k = shape
        sgblas_mean = 1.0 + factor + (m + n + k) / 1.0e7
        cublas_mean = sgblas_mean * 0.8
        operations = 2.0 * m * n * k
        sgblas_gflops = operations / (sgblas_mean * 1.0e6)
        cublas_gflops = operations / (cublas_mean * 1.0e6)
        ratio = sgblas_gflops / cublas_gflops
        values = {
            "sgblas_mean_ms": sgblas_mean,
            "sgblas_gflops": sgblas_gflops,
            "cublas_mean_ms": cublas_mean,
            "cublas_gflops": cublas_gflops,
            "ratio": ratio,
        }
        record: dict[str, object] = {
            "record_type": "result",
            "schema_version": 1,
            "timed_order": order,
            "warmups_per_implementation": warmups,
            "timed_repeats_per_implementation": repeats,
            "m": m,
            "n": n,
            "k": k,
            "sgblas_total_ms": sgblas_mean * repeats,
            "sgblas_mean_ms": sgblas_mean,
            "sgblas_gflops": sgblas_gflops,
            "cublas_total_ms": cublas_mean * repeats,
            "cublas_mean_ms": cublas_mean,
            "cublas_gflops": cublas_gflops,
            "ratio": ratio,
        }
        return record, values

    def _benchmark_body(
        self,
        argv: list[str],
        order: str,
        shapes: tuple[tuple[int, int, int], ...],
        run_index: int,
        *,
        warmups: int = 10,
        repeats: int = 100,
    ) -> tuple[str, dict[tuple[int, int, int], dict[str, float]]]:
        records: list[dict[str, object]] = [
            self._metadata(argv, order, warmups, repeats, 20260711)
        ]
        values: dict[tuple[int, int, int], dict[str, float]] = {}
        for shape_index, shape in enumerate(shapes):
            record, result_values = self._result(
                shape,
                order,
                warmups,
                repeats,
                run_index / 1000.0 + shape_index / 10000.0,
            )
            records.append(record)
            values[shape] = result_values
        return "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", values

    def _make_prime(self) -> None:
        for variant in verify.VARIANTS:
            order = self.prime_order[variant]
            argv = [
                str(self.binaries[variant]["benchmark"]),
                "4096",
                "4096",
                "4096",
                "--warmups",
                "10",
                "--repeats",
                "20",
                "--seed",
                "20260711",
                "--order",
                order,
                "--output",
                "jsonl",
            ]
            body, _ = self._benchmark_body(
                argv,
                order,
                ((4096, 4096, 4096),),
                0,
                warmups=10,
                repeats=20,
            )
            self._logged(
                self.run / variant / "prime.log",
                argv,
                body,
                environment="benchmark",
            )

    def _make_benchmarks(self, balanced: bool, missing_shape: bool) -> None:
        shapes = verify.DEFAULT_SHAPES[:-1] if missing_shape else verify.DEFAULT_SHAPES
        for run_index, process_order in enumerate(self.full_order, start=1):
            for variant in process_order:
                order = (
                    "sgblas-first"
                    if not balanced
                    else self.timing_full[variant][run_index - 1]
                )
                argv = [
                    str(self.binaries[variant]["benchmark"]),
                    "--warmups",
                    "10",
                    "--repeats",
                    "100",
                    "--seed",
                    "20260711",
                    "--order",
                    order,
                    "--output",
                    "jsonl",
                ]
                body, values = self._benchmark_body(
                    argv, order, shapes, run_index
                )
                self._logged(
                    self.run / variant / f"full-{run_index:02d}.log",
                    argv,
                    body,
                    environment="benchmark",
                )
                self.values[variant].append(values)

    def _make_sanitizers(self, bad_sanitizer: str | None) -> None:
        for tool in verify.SANITIZER_TOOLS:
            argv = [
                "/usr/local/cuda/bin/compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode=99",
                str(self.binaries["hybrid"]["correctness"]),
            ]
            if tool == "racecheck":
                summary = (
                    "========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)"
                    if bad_sanitizer == tool
                    else "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                )
            else:
                count = 1 if bad_sanitizer == tool else 0
                summary = f"========= ERROR SUMMARY: {count} errors"
            body = (
                "========= COMPUTE-SANITIZER\n"
                + self._correctness_text(True)
                + summary
                + "\n"
            )
            self._logged(
                self.run / "hybrid" / f"sanitizer-{tool}.log", argv, body
            )

    def _make_summary(self) -> None:
        summary: dict[str, object] = {}
        for variant in verify.VARIANTS:
            shapes = tuple(self.values[variant][0])
            rows: list[dict[str, object]] = []
            for shape in shapes:
                samples = [run[shape] for run in self.values[variant]]
                rows.append(
                    {
                        "m": shape[0],
                        "n": shape[1],
                        "k": shape[2],
                        "runs": len(samples),
                        "sgblas_ms": statistics.median(
                            sample["sgblas_mean_ms"] for sample in samples
                        ),
                        "sgblas_gflops": statistics.median(
                            sample["sgblas_gflops"] for sample in samples
                        ),
                        "sgblas_gflops_min": min(
                            sample["sgblas_gflops"] for sample in samples
                        ),
                        "sgblas_gflops_max": max(
                            sample["sgblas_gflops"] for sample in samples
                        ),
                        "cublas_ms": statistics.median(
                            sample["cublas_mean_ms"] for sample in samples
                        ),
                        "cublas_gflops": statistics.median(
                            sample["cublas_gflops"] for sample in samples
                        ),
                        "ratio": statistics.median(sample["ratio"] for sample in samples),
                    }
                )
            large_ratios = [
                row["ratio"]
                for row in rows
                if (row["m"], row["n"], row["k"]) in verify.DEFAULT_SHAPES[-4:]
            ]
            summary[variant] = {
                "rows": rows,
                "large_shape_geometric_mean": math.prod(large_ratios)
                ** (1.0 / len(large_ratios)),
            }
            write(self.run / f"summary-{variant}.md", f"# {variant} synthetic summary\n")
        write(self.run / "summary.json", json.dumps(summary, indent=2) + "\n")

    @staticmethod
    def _telemetry_gpu(index: int) -> dict[str, object]:
        return {
            "reported_timestamp": f"2026/08/20 00:10:{index % 60:02d}.000",
            "name": "NVIDIA A100-SXM4-80GB",
            "driver_version": "580.0",
            "performance_state": "P0",
            "temperature_c": 42,
            "power_draw_w": 180.5,
            "power_limit_w": 400.0,
            "graphics_clock_mhz": 1410,
            "sm_clock_mhz": 1410,
            "memory_clock_mhz": 1593,
            "clock_throttle_reasons_active": "0x0000000000000000",
            "gpu_utilization_percent": 0,
            "memory_utilization_percent": 0,
            "memory_used_mib": 0,
            "memory_total_mib": 81920,
            "compute_mode": "Default",
            "mig_mode": "Disabled",
        }

    def _telemetry_records(self) -> list[dict[str, object]]:
        schedule: list[tuple[str, str, str, int]] = [
            ("before", "prime", variant, 1) for variant in verify.VARIANTS
        ]
        for run_index, process_order in enumerate(self.full_order, start=1):
            for variant in process_order:
                schedule.extend(
                    (
                        ("before", "full", variant, run_index),
                        ("after", "full", variant, run_index),
                    )
                )
        records: list[dict[str, object]] = []
        for index, (phase, workload, variant, run_index) in enumerate(schedule):
            second = 120 + index * 2
            started = dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc) + dt.timedelta(
                seconds=second
            )
            records.append(
                {
                    "phase": phase,
                    "workload": workload,
                    "variant": variant,
                    "run_index": run_index,
                    "started_utc": started.isoformat(timespec="milliseconds").replace(
                        "+00:00", "Z"
                    ),
                    "finished_utc": (started + dt.timedelta(seconds=1))
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    "gpu": self._telemetry_gpu(index),
                    "active_compute_processes": [],
                }
            )
        return records

    def _make_manifest(self) -> dict[str, object]:
        git_state = {
            "head": self.commit,
            "status_porcelain": "",
            "available": True,
            "dirty": False,
        }
        return {
            "schema_version": 1,
            "state": "complete",
            "started_utc": "2026-08-20T00:00:00.000Z",
            "finished_utc": "2026-08-20T01:00:00.000Z",
            "duration_seconds": 3600.0,
            "failure": None,
            "container": {
                "image": "sgblas-a100:fixture",
                "image_digest": "sha256:" + "a" * 64,
            },
            "source": {
                "path": str(self.source),
                "sha256_before": self.source_digest,
                "sha256_after": self.source_digest,
                "files": self.source_files,
                "git_before": git_state,
                "git_after": copy.deepcopy(git_state),
            },
            "options": {
                "build_root": str(self.build),
                "results_root": str(self.run.parent),
                "runs": self.runs,
                "warmups": 10,
                "repeats": 100,
                "jobs": 2,
                "seed": 20260711,
                "include_8192": False,
                "repeats_8192": None,
                "sanitizers": True,
                "command_timeout_seconds": 1800,
                "total_timeout_seconds": 7200,
                "tf32_override": "0",
                "full_order": self.full_order,
                "order_8192": None,
                "timing_order_prime": self.prime_order,
                "timing_order_full": self.timing_full,
                "timing_order_8192": None,
            },
            "variants": copy.deepcopy(verify.EXPECTED_VARIANTS),
            "child_environments": {
                "base": {
                    "LANG": "C",
                    "LC_ALL": "C",
                    "TZ": "UTC",
                    "TMPDIR": "/tmp",
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "SGBLAS_CONTAINER_IMAGE": "sgblas-a100:fixture",
                    "SGBLAS_CONTAINER_IMAGE_DIGEST": "sha256:" + "a" * 64,
                },
                "benchmark": {
                    "LANG": "C",
                    "LC_ALL": "C",
                    "TZ": "UTC",
                    "TMPDIR": "/tmp",
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "SGBLAS_CONTAINER_IMAGE": "sgblas-a100:fixture",
                    "SGBLAS_CONTAINER_IMAGE_DIGEST": "sha256:" + "a" * 64,
                    "NVIDIA_TF32_OVERRIDE": "0",
                },
            },
            "commands": self.commands,
            "probe": {
                "nvidia_smi": "NVIDIA A100-SXM4-80GB, 580.0, 8.0, 81920 MiB",
                "gpu_identity": {
                    "name": "NVIDIA A100-SXM4-80GB",
                    "driver_version": "580.0",
                    "compute_capability": "8.0",
                    "memory_total_mib": 81920,
                },
                "nvidia_smi_q": (
                    "Product Name : NVIDIA A100-SXM4-80GB\n"
                    "GPU UUID : <redacted>\nSerial Number : <redacted>\n"
                ),
                "nvidia_smi_q_identifiers_redacted": True,
                "active_compute_processes": [],
                "nvcc": "Cuda compilation tools, release 12.8, V12.8.1",
                "cmake": "cmake version 4.0.0",
                "uname_a": "Linux fixture 6.8.0 x86_64 GNU/Linux",
                "os_release": "NAME=Fixture Linux\nVERSION_ID=1\n",
                "os_release_sha256": hashlib.sha256(
                    b"NAME=Fixture Linux\nVERSION_ID=1\n"
                ).hexdigest(),
                "python": "3.13.0",
            },
            "telemetry": self._telemetry_records(),
            "tools": {
                "nvidia_smi": "/usr/bin/nvidia-smi",
                "nvcc": "/usr/local/cuda/bin/nvcc",
                "cmake": "/usr/local/bin/cmake",
                "git": "/usr/bin/git",
                "uname": "/usr/bin/uname",
                "compute_sanitizer": "/usr/local/cuda/bin/compute-sanitizer",
            },
            "binaries": {
                variant: {
                    kind: artifact_file_record(
                        self.copied_binaries[variant][kind], self.run
                    )
                    for kind in ("correctness", "benchmark")
                }
                for variant in verify.VARIANTS
            },
            "binary_copy_proofs": {
                variant: {
                    kind: copy_proof(
                        self.binaries[variant][kind],
                        self.copied_binaries[variant][kind],
                        self.run,
                    )
                    for kind in ("correctness", "benchmark")
                }
                for variant in verify.VARIANTS
            },
            "build_evidence": {
                variant: {
                    "cmake_cache": copy_proof(
                        self.cmake_caches[variant][0],
                        self.cmake_caches[variant][1],
                        self.run,
                    ),
                    "configure_log": artifact_file_record(
                        self.run / variant / "configure.log", self.run
                    ),
                    "build_log": artifact_file_record(
                        self.run / variant / "build.log", self.run
                    ),
                }
                for variant in verify.VARIANTS
            },
            "compiler_spill_reports": {
                variant: [{"stores_bytes": 0, "loads_bytes": 0}]
                for variant in verify.VARIANTS
            },
            "artifacts": {},
            "manifest_excluded_from_artifact_hashes": True,
        }

    def _finalize_inventory(self) -> None:
        inventory = {
            path.relative_to(self.run).as_posix(): artifact_file_record(path, self.run)
            for path in sorted(candidate for candidate in self.run.rglob("*") if candidate.is_file())
        }
        self.manifest["artifacts"] = inventory
        write(self.run / "manifest.json", json.dumps(self.manifest, indent=2) + "\n")

    def rewrite_manifest(self) -> None:
        write(self.run / "manifest.json", json.dumps(self.manifest, indent=2) + "\n")


class VerifyEvidenceTests(unittest.TestCase):
    def make_fixture(self, **kwargs: object) -> tuple[tempfile.TemporaryDirectory[str], Fixture]:
        temporary = tempfile.TemporaryDirectory()
        fixture = Fixture(Path(temporary.name), **kwargs)
        return temporary, fixture

    def test_valid_bundle_passes(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        report = verify.verify_evidence(fixture.run)
        self.assertEqual(report["independent_logs_per_variant"], 6)
        self.assertEqual(report["git_commit"], fixture.commit)

    def test_incomplete_manifest_fails(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.manifest["state"] = "incomplete"
        fixture.rewrite_manifest()
        with self.assertRaisesRegex(verify.VerificationError, "not complete"):
            verify.verify_evidence(fixture.run)

    def test_legacy_uuid_probe_schema_fails(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.manifest["probe"]["nvidia_smi"] = (
            "NVIDIA A100-SXM4-80GB, GPU-fixture, 580.0, 8.0, 81920 MiB"
        )
        fixture.rewrite_manifest()
        with self.assertRaisesRegex(verify.VerificationError, "name, driver"):
            verify.verify_evidence(fixture.run)

    def test_invalid_probe_driver_fails(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.manifest["probe"]["nvidia_smi"] = (
            "NVIDIA A100-SXM4-80GB, unknown, 8.0, 81920 MiB"
        )
        fixture.rewrite_manifest()
        with self.assertRaisesRegex(verify.VerificationError, "driver version"):
            verify.verify_evidence(fixture.run)

    def test_source_hash_mismatch_fails(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        write(fixture.source / "src" / "kernel.cu", "// tampered\n")
        with self.assertRaisesRegex(verify.VerificationError, "source.files|checksum"):
            verify.verify_evidence(fixture.run)

    def test_binary_hash_mismatch_fails(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        write(fixture.copied_binaries["hybrid"]["benchmark"], "tampered binary\n")
        with self.assertRaisesRegex(verify.VerificationError, "checksum mismatch|size mismatch"):
            verify.verify_evidence(fixture.run)

    def test_relocated_bundle_uses_copied_binaries_and_source_override(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        relocated_run = fixture.root / "relocated" / "run"
        relocated_source = fixture.root / "relocated" / "source"
        shutil.copytree(fixture.run, relocated_run)
        shutil.copytree(fixture.source, relocated_source)
        shutil.rmtree(fixture.run)
        shutil.rmtree(fixture.source)
        shutil.rmtree(fixture.build)
        report = verify.verify_evidence(relocated_run, relocated_source)
        self.assertEqual(report["git_commit"], fixture.commit)

    def test_artifact_checksum_mismatch_fails(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        path = fixture.run / "wide" / "full-01.log"
        path.write_text(path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(verify.VerificationError, "checksum mismatch|size mismatch"):
            verify.verify_evidence(fixture.run)

    def test_insufficient_independent_logs_fail(self) -> None:
        temporary, fixture = self.make_fixture(runs=4)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(verify.VerificationError, "even and at least 6"):
            verify.verify_evidence(fixture.run)

    def test_fewer_than_ten_warmups_fails(self) -> None:
        temporary, fixture = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        fixture.manifest["options"]["warmups"] = 9
        fixture.rewrite_manifest()
        with self.assertRaisesRegex(verify.VerificationError, "warmups must be at least 10"):
            verify.verify_evidence(fixture.run)

    def test_benchmark_metadata_below_ten_warmups_fails(self) -> None:
        argv = ["/tmp/sgblas_benchmark"]
        metadata = Fixture._metadata(argv, "sgblas-first", 9, 100, 20260711)
        with self.assertRaisesRegex(verify.VerificationError, "fewer than 10 warmups"):
            verify.validate_benchmark_contract(
                metadata,
                argv,
                "sgblas-first",
                9,
                100,
                20260711,
                "synthetic benchmark",
            )

    def test_unbalanced_library_order_fails(self) -> None:
        temporary, fixture = self.make_fixture(balanced=False)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(
            verify.VerificationError, "timing order is unbalanced|differs from timing schedule"
        ):
            verify.verify_evidence(fixture.run)

    def test_missing_shape_fails(self) -> None:
        temporary, fixture = self.make_fixture(missing_shape=True)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(verify.VerificationError, "result count mismatch|shape corpus"):
            verify.verify_evidence(fixture.run)

    def test_bad_sanitizer_summary_fails(self) -> None:
        temporary, fixture = self.make_fixture(bad_sanitizer="racecheck")
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(verify.VerificationError, "racecheck clean summary missing"):
            verify.verify_evidence(fixture.run)

    def test_missing_correctness_pass_fails(self) -> None:
        temporary, fixture = self.make_fixture(bad_correctness=True)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(verify.VerificationError, "correctness-pass marker"):
            verify.verify_evidence(fixture.run)


if __name__ == "__main__":
    unittest.main()
