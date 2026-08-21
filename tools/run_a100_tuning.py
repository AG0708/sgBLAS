#!/usr/bin/env python3
"""Run a reproducible wide-versus-hybrid sgBLAS tuning campaign on A100."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from statistics import median


DEFAULT_SHAPES = (
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 1024, 4096),
    (1024, 4096, 4096),
)
LARGE_SHAPES = set(DEFAULT_SHAPES[-4:])
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
CONTAINER_ENV_KEYS = (
    "SGBLAS_CONTAINER_IMAGE",
    "SGBLAS_CONTAINER_IMAGE_DIGEST",
)
TELEMETRY_FIELDS = (
    "timestamp",
    "name",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks_throttle_reasons.active",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "compute_mode",
    "mig.mode.current",
)
COMPUTE_PROCESS_FIELDS = ("pid", "used_gpu_memory")
SOURCE_ROOTS = (
    ".dockerignore",
    "CMakeLists.txt",
    "CMakePresets.json",
    "LICENSE",
    "PROVENANCE.md",
    "THIRD_PARTY_NOTICES.md",
    "bench",
    "cmake",
    "include",
    "infra/runpod",
    "src",
    "tests",
    "tools",
)
SPILL_PATTERN = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
ENV_ALLOWLIST = (
    "CC",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_MODULE_LOADING",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDAHOSTCXX",
    "CXX",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PKG_CONFIG_PATH",
    *CONTAINER_ENV_KEYS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, validate, and repeatedly benchmark the SM80 wide and "
            "shape-dispatched hybrid kernels. Run this on the A100 host."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="sgBLAS source tree (default: repository root)",
    )
    parser.add_argument("--build-root", type=Path, default=Path("build/tuning"))
    parser.add_argument("--results-root", type=Path, default=Path("results/tuning"))
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=1800,
        help="hard timeout for any one child process (default: 1800)",
    )
    parser.add_argument(
        "--total-timeout-seconds",
        type=int,
        default=7200,
        help="hard deadline for the complete campaign (default: 7200)",
    )
    parser.add_argument(
        "--include-8192",
        action="store_true",
        help="also run each variant on 8192 cubed with 50 repeats",
    )
    parser.add_argument(
        "--sanitizers",
        action="store_true",
        help="run all four Compute Sanitizer tools on the hybrid candidate",
    )
    args = parser.parse_args()
    for name in (
        "runs",
        "warmups",
        "repeats",
        "jobs",
        "command_timeout_seconds",
        "total_timeout_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.runs < 6 or args.runs % 2:
        parser.error("--runs must be an even number of at least 6 for balanced ordering")
    if args.warmups < 10:
        parser.error("--warmups must be at least 10")
    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must be between 0 and 4294967295")
    return args


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def balanced_timing_orders(runs: int) -> dict[str, list[str]]:
    if runs <= 0 or runs % 2:
        raise ValueError("balanced timing orders require a positive even run count")
    schedules = {
        name: [
            "sgblas-first" if (index + offset) % 2 == 0 else "cublas-first"
            for index in range(runs)
        ]
        for name, offset in (("wide", 0), ("hybrid", 1))
    }
    for name, schedule in schedules.items():
        if schedule.count("sgblas-first") != runs // 2:
            raise RuntimeError(f"unbalanced timing order for {name}")
    return schedules


def validate_gpu_query(output: str) -> dict[str, object]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {len(lines)}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi result: {lines[0]}")
    name, driver_version, compute_capability, memory_total = fields
    if re.fullmatch(r"\d+(?:\.\d+){1,3}", driver_version) is None:
        raise RuntimeError(f"invalid NVIDIA driver version: {driver_version}")
    memory_match = re.fullmatch(r"(\d+)\s+MiB", memory_total)
    if (
        name != EXPECTED_GPU_NAME
        or compute_capability != "8.0"
        or memory_match is None
        or int(memory_match.group(1)) < 80000
    ):
        raise RuntimeError(
            f"this campaign requires exactly {EXPECTED_GPU_NAME} with compute 8.0"
        )
    return {
        "name": name,
        "driver_version": driver_version,
        "compute_capability": compute_capability,
        "memory_total_mib": int(memory_match.group(1)),
    }


def container_provenance() -> dict[str, str]:
    values: dict[str, str] = {}
    for key in CONTAINER_ENV_KEYS:
        value = os.environ.get(key)
        if value is None or not value.strip():
            raise RuntimeError(f"required controller environment is missing: {key}")
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise RuntimeError(f"invalid controller environment value: {key}")
        values[key] = value
    return {
        "image": values["SGBLAS_CONTAINER_IMAGE"],
        "image_digest": values["SGBLAS_CONTAINER_IMAGE_DIGEST"],
    }


def _integer_field(value: str, name: str) -> int:
    if re.fullmatch(r"\d+", value) is None:
        raise RuntimeError(f"invalid telemetry {name}: {value}")
    return int(value)


def _float_field(value: str, name: str) -> float:
    if re.fullmatch(r"\d+(?:\.\d+)?", value) is None:
        raise RuntimeError(f"invalid telemetry {name}: {value}")
    return float(value)


def parse_gpu_telemetry(output: str) -> dict[str, object]:
    rows = [row for row in csv.reader(output.splitlines()) if any(field.strip() for field in row)]
    if len(rows) != 1 or len(rows[0]) != len(TELEMETRY_FIELDS):
        raise RuntimeError("structured GPU telemetry must contain exactly one complete row")
    values = {
        key: value.strip() for key, value in zip(TELEMETRY_FIELDS, rows[0])
    }
    if any(not value for value in values.values()):
        raise RuntimeError("structured GPU telemetry contains an empty field")
    if values["name"] != EXPECTED_GPU_NAME:
        raise RuntimeError(f"telemetry GPU is not exactly {EXPECTED_GPU_NAME}")
    if re.fullmatch(r"\d+(?:\.\d+){1,3}", values["driver_version"]) is None:
        raise RuntimeError("telemetry driver version is invalid")
    if re.fullmatch(r"P\d+", values["pstate"]) is None:
        raise RuntimeError("telemetry performance state is invalid")
    result: dict[str, object] = {
        "reported_timestamp": values["timestamp"],
        "name": values["name"],
        "driver_version": values["driver_version"],
        "performance_state": values["pstate"],
        "temperature_c": _integer_field(values["temperature.gpu"], "temperature"),
        "power_draw_w": _float_field(values["power.draw"], "power.draw"),
        "power_limit_w": _float_field(values["power.limit"], "power.limit"),
        "graphics_clock_mhz": _integer_field(
            values["clocks.current.graphics"], "graphics clock"
        ),
        "sm_clock_mhz": _integer_field(values["clocks.current.sm"], "SM clock"),
        "memory_clock_mhz": _integer_field(
            values["clocks.current.memory"], "memory clock"
        ),
        "clock_throttle_reasons_active": values[
            "clocks_throttle_reasons.active"
        ],
        "gpu_utilization_percent": _integer_field(
            values["utilization.gpu"], "GPU utilization"
        ),
        "memory_utilization_percent": _integer_field(
            values["utilization.memory"], "memory utilization"
        ),
        "memory_used_mib": _integer_field(values["memory.used"], "memory used"),
        "memory_total_mib": _integer_field(values["memory.total"], "memory total"),
        "compute_mode": values["compute_mode"],
        "mig_mode": values["mig.mode.current"],
    }
    if (
        int(result["memory_total_mib"]) < 80000
        or not 0 <= int(result["memory_used_mib"]) <= int(
            result["memory_total_mib"]
        )
        or not 0 < int(result["temperature_c"]) < 120
        or float(result["power_draw_w"]) < 0.0
        or float(result["power_limit_w"]) <= 0.0
        or int(result["graphics_clock_mhz"]) <= 0
        or int(result["sm_clock_mhz"]) <= 0
        or int(result["memory_clock_mhz"]) <= 0
        or not 0 <= int(result["gpu_utilization_percent"]) <= 100
        or not 0 <= int(result["memory_utilization_percent"]) <= 100
    ):
        raise RuntimeError("structured GPU telemetry is outside valid bounds")
    return result


def parse_compute_processes(output: str) -> list[dict[str, int | None]]:
    stripped = output.strip()
    if not stripped or stripped == "No running processes found":
        return []
    processes: list[dict[str, int | None]] = []
    for row in csv.reader(stripped.splitlines()):
        fields = [field.strip() for field in row]
        if len(fields) != 2 or re.fullmatch(r"\d+", fields[0]) is None:
            raise RuntimeError(f"invalid active compute-process row: {row}")
        memory = None
        if fields[1] not in ("N/A", "[Not Supported]"):
            memory = _integer_field(fields[1], "process memory")
        pid = int(fields[0])
        if pid <= 0 or any(process["pid"] == pid for process in processes):
            raise RuntimeError(f"invalid or duplicate compute-process PID: {pid}")
        processes.append({"pid": pid, "used_gpu_memory_mib": memory})
    return processes


def redact_nvidia_smi_q(output: str) -> str:
    redacted = re.sub(
        r"(?im)^(\s*(?:GPU UUID|Serial Number)\s*:\s*).+$",
        r"\1<redacted>",
        output,
    )
    return re.sub(r"\bGPU-[0-9A-Fa-f-]{16,}\b", "GPU-<redacted>", redacted)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_relative_path(path: Path, run_root: Path) -> str:
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"artifact path escapes run root: {path}") from error


def artifact_file_record(path: Path, run_root: Path) -> dict[str, object]:
    record = file_record(path)
    record["path"] = artifact_relative_path(path, run_root)
    return record


def artifact_copy_proof(
    source: Path, destination: Path, run_root: Path
) -> dict[str, object]:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"evidence source is not a regular file: {source}")
    if not destination.is_file() or destination.is_symlink():
        raise RuntimeError(f"copied evidence is not a regular file: {destination}")
    original = file_record(source)
    copied = artifact_file_record(destination, run_root)
    hashes_equal = original["sha256"] == copied["sha256"]
    sizes_equal = original["size_bytes"] == copied["size_bytes"]
    if not hashes_equal or not sizes_equal:
        raise RuntimeError(f"copied evidence differs from source: {source}")
    return {
        "original": original,
        "copied": copied,
        "sha256_equal": hashes_equal,
        "size_equal": sizes_equal,
    }


def copy_verified_artifact(
    source: Path, destination: Path, run_root: Path
) -> dict[str, object]:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"evidence source is not a regular file: {source}")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"evidence destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return artifact_copy_proof(source, destination, run_root)


def revalidate_artifact_copy(
    source: Path,
    destination: Path,
    run_root: Path,
    expected: dict[str, object],
) -> None:
    if artifact_copy_proof(source, destination, run_root) != expected:
        raise RuntimeError(f"evidence copy changed after capture: {destination}")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def child_environment() -> dict[str, str]:
    environment = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    environment.setdefault(
        "PATH",
        "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    )
    environment.update({"LANG": "C", "LC_ALL": "C", "TMPDIR": "/tmp", "TZ": "UTC"})
    return environment


class CommandRunner:
    def __init__(
        self,
        *,
        deadline: float,
        command_timeout: int,
        artifact_root: Path,
        environments: dict[str, dict[str, str]],
        commands: list[dict[str, object]],
        persist: Callable[[], None],
    ) -> None:
        self.deadline = deadline
        self.command_timeout = command_timeout
        self.artifact_root = artifact_root
        self.environments = environments
        self.commands = commands
        self.persist = persist

    def ensure_time(self, operation: str) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"total campaign deadline exhausted before {operation}")
        return min(float(self.command_timeout), remaining)

    def _begin(
        self, command: list[str], cwd: Path, environment: str, log_path: Path | None
    ) -> tuple[dict[str, object], float]:
        record: dict[str, object] = {
            "argv": command,
            "cwd": str(cwd),
            "environment": environment,
            "started_utc": utc_now(),
            "finished_utc": None,
            "duration_seconds": None,
            "state": "running",
            "exit_code": None,
        }
        if log_path is not None:
            record["log_path"] = artifact_relative_path(
                log_path, self.artifact_root
            )
        self.commands.append(record)
        self.persist()
        return record, time.monotonic()

    @staticmethod
    def _stop(process: subprocess.Popen[str]) -> str:
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
        return output or ""

    def _finish(
        self,
        record: dict[str, object],
        started: float,
        process: subprocess.Popen[str] | None,
        state: str,
        error: BaseException | None,
    ) -> None:
        record.update(
            {
                "finished_utc": utc_now(),
                "duration_seconds": round(time.monotonic() - started, 6),
                "state": state,
                "exit_code": None if process is None else process.returncode,
            }
        )
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        self.persist()

    def capture(
        self,
        command: list[str],
        cwd: Path,
        *,
        environment: str = "base",
        allow_failure: bool = False,
    ) -> str:
        timeout = self.ensure_time(command[0])
        record, started = self._begin(command, cwd, environment, None)
        process: subprocess.Popen[str] | None = None
        output = ""
        error: BaseException | None = None
        state = "failed"
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=self.environments[environment],
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            try:
                output, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                output = self._stop(process)
                error = RuntimeError(
                    f"command exceeded {timeout:.1f}s timeout: {' '.join(command)}"
                )
                state = "timeout"
            except BaseException as caught:
                output = self._stop(process)
                error = caught
                state = "interrupted"
            else:
                state = "complete" if process.returncode == 0 else "failed"
        except BaseException as caught:
            error = caught
        record["output_sha256"] = hashlib.sha256(output.encode("utf-8")).hexdigest()
        self._finish(record, started, process, state, error)
        if error is not None:
            raise error
        if process is None or (process.returncode != 0 and not allow_failure):
            raise RuntimeError(
                f"command failed with status {None if process is None else process.returncode}: "
                + " ".join(command)
            )
        return output.strip()

    def logged(
        self,
        command: list[str],
        cwd: Path,
        log_path: Path,
        *,
        environment: str = "base",
    ) -> None:
        timeout = self.ensure_time(command[0])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record, started = self._begin(command, cwd, environment, log_path)
        process: subprocess.Popen[str] | None = None
        error: BaseException | None = None
        state = "failed"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("# command: " + json.dumps(command) + "\n")
                log.write(f"# started_utc: {record['started_utc']}\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=self.environments[environment],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name == "posix"),
                )
                try:
                    process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._stop(process)
                    error = RuntimeError(
                        f"command exceeded {timeout:.1f}s timeout: {' '.join(command)}"
                    )
                    state = "timeout"
                except BaseException as caught:
                    self._stop(process)
                    error = caught
                    state = "interrupted"
                else:
                    state = "complete" if process.returncode == 0 else "failed"
                log.write(f"# finished_utc: {utc_now()}\n")
                log.write(f"# exit_code: {process.returncode}\n")
        except BaseException as caught:
            if process is not None and process.poll() is None:
                self._stop(process)
            if error is None:
                error = caught
        if log_path.is_file():
            record["log"] = artifact_file_record(log_path, self.artifact_root)
        self._finish(record, started, process, state, error)
        if log_path.is_file():
            sys.stdout.write(log_path.read_text(encoding="utf-8", errors="replace"))
        if error is not None:
            raise error
        if process is None or process.returncode != 0:
            raise RuntimeError(
                f"command failed with status {None if process is None else process.returncode}; "
                f"see {log_path}: " + " ".join(command)
            )


def find_tool(name: str, fallback: str | None = None) -> str:
    path = shutil.which(name)
    if path is not None:
        return path
    if fallback is not None and Path(fallback).is_file():
        return fallback
    raise RuntimeError(f"required tool not found: {name}")


def source_snapshot(source: Path) -> tuple[str, dict[str, dict[str, object]]]:
    digest = hashlib.sha256()
    files: list[Path] = []
    for entry in SOURCE_ROOTS:
        path = source / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise RuntimeError(f"required source/provenance input is missing: {path}")
    records: dict[str, dict[str, object]] = {}
    for path in sorted(files, key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        contents = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
        records[relative] = {
            "size_bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
    return digest.hexdigest(), records


def parse_benchmark_jsonl(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metadata_records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL in {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise RuntimeError(f"non-object JSONL record in {path}:{line_number}")
        if record.get("record_type") == "metadata":
            metadata_records.append(record)
            continue
        if record.get("record_type") != "result":
            raise RuntimeError(f"unknown JSONL record in {path}:{line_number}")
        dimensions: list[int] = []
        for key in ("m", "n", "k"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"invalid {key} in {path}:{line_number}")
            dimensions.append(value)
        values: list[float] = []
        for key in (
            "sgblas_mean_ms",
            "sgblas_gflops",
            "cublas_mean_ms",
            "cublas_gflops",
            "ratio",
        ):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"missing numeric {key} in {path}:{line_number}")
            converted = float(value)
            if not math.isfinite(converted) or converted <= 0.0:
                raise RuntimeError(f"invalid {key} in {path}:{line_number}")
            values.append(converted)
        rows.append(
            {
                "m": dimensions[0],
                "n": dimensions[1],
                "k": dimensions[2],
                "sgblas_ms": values[0],
                "sgblas_gflops": values[1],
                "cublas_ms": values[2],
                "cublas_gflops": values[3],
                "ratio": values[4],
                "timed_order": record.get("timed_order", ""),
                "warmups": record.get("warmups_per_implementation", -1),
                "repeats": record.get("timed_repeats_per_implementation", -1),
                "schema_version": record.get("schema_version", -1),
            }
        )
    if len(metadata_records) != 1:
        raise RuntimeError(
            f"expected exactly one JSONL metadata record in {path}, got {len(metadata_records)}"
        )
    if not rows:
        raise RuntimeError(f"no full-precision JSONL result records in {path}")
    return metadata_records[0], rows


def parse_rows(path: Path) -> list[dict[str, object]]:
    return parse_benchmark_jsonl(path)[1]


def validate_log(
    path: Path,
    expected_shapes: tuple[tuple[int, int, int], ...],
    *,
    expected_order: str,
    expected_warmups: int,
    expected_repeats: int,
    expected_seed: int,
    expected_argv: list[str],
) -> None:
    metadata, rows = parse_benchmark_jsonl(path)
    benchmark = metadata.get("benchmark")
    device = metadata.get("device")
    if metadata.get("schema_version") != 1 or not isinstance(benchmark, dict):
        raise RuntimeError(f"invalid benchmark metadata schema in {path}")
    if not isinstance(device, dict):
        raise RuntimeError(f"missing device metadata in {path}")
    expected_metadata = {
        "timed_order": expected_order,
        "warmup_order": "alternating-per-launch",
        "warmups_per_implementation": expected_warmups,
        "timed_repeats_per_implementation": expected_repeats,
        "seed": expected_seed,
        "cache_policy": "same-buffer-steady-state",
        "stream": "shared-nonblocking",
        "sgblas_math_mode": "SGBLAS_MATH_FP32",
        "cublas_math_mode": "CUBLAS_PEDANTIC_MATH",
        "cublas_compute_type": "CUBLAS_COMPUTE_32F_PEDANTIC",
        "cublas_algorithm": "CUBLAS_GEMM_DEFAULT",
        "nvidia_tf32_override": "0",
    }
    for key, expected in expected_metadata.items():
        if benchmark.get(key) != expected:
            raise RuntimeError(
                f"metadata mismatch for {key} in {path}: "
                f"expected {expected!r}, got {benchmark.get(key)!r}"
            )
    if metadata.get("argv") != expected_argv:
        raise RuntimeError(f"benchmark argv mismatch in {path}")
    if (
        device.get("name") != EXPECTED_GPU_NAME
        or device.get("compute_capability_major") != 8
        or device.get("compute_capability_minor") != 0
    ):
        raise RuntimeError(f"unexpected benchmark device metadata in {path}")
    actual = tuple((int(row["m"]), int(row["n"]), int(row["k"])) for row in rows)
    if actual != expected_shapes:
        raise RuntimeError(
            f"unexpected benchmark rows in {path}: expected {expected_shapes}, got {actual}"
        )
    for row in rows:
        if (
            row["schema_version"] != 1
            or row["timed_order"] != expected_order
            or row["warmups"] != expected_warmups
            or row["repeats"] != expected_repeats
        ):
            raise RuntimeError(f"result metadata mismatch in {path}")


def summarize(
    logs: list[Path], expected_counts: dict[tuple[int, int, int], int]
) -> dict[str, object]:
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for path in logs:
        for row in parse_rows(path):
            shape = (int(row["m"]), int(row["n"]), int(row["k"]))
            grouped.setdefault(shape, []).append(row)

    if set(grouped) != set(expected_counts):
        raise RuntimeError(
            f"summary shape mismatch: expected {sorted(expected_counts)}, got {sorted(grouped)}"
        )
    for shape, count in expected_counts.items():
        if len(grouped[shape]) != count:
            raise RuntimeError(
                f"summary run-count mismatch for {shape}: expected {count}, "
                f"got {len(grouped[shape])}"
            )
        orders = [row.get("timed_order") for row in grouped[shape]]
        if (
            orders.count("sgblas-first") != count // 2
            or orders.count("cublas-first") != count // 2
        ):
            raise RuntimeError(f"summary timing-order imbalance for {shape}")

    result_rows: list[dict[str, float | int]] = []
    for shape, rows in grouped.items():
        result_rows.append(
            {
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "runs": len(rows),
                "sgblas_ms": median(float(row["sgblas_ms"]) for row in rows),
                "sgblas_gflops": median(
                    float(row["sgblas_gflops"]) for row in rows
                ),
                "sgblas_gflops_min": min(
                    float(row["sgblas_gflops"]) for row in rows
                ),
                "sgblas_gflops_max": max(
                    float(row["sgblas_gflops"]) for row in rows
                ),
                "cublas_ms": median(float(row["cublas_ms"]) for row in rows),
                "cublas_gflops": median(
                    float(row["cublas_gflops"]) for row in rows
                ),
                "ratio": median(float(row["ratio"]) for row in rows),
            }
        )
    result_rows.sort(key=lambda row: (int(row["m"]), int(row["n"]), int(row["k"])))
    large_ratios = [
        float(row["ratio"])
        for row in result_rows
        if (int(row["m"]), int(row["n"]), int(row["k"])) in LARGE_SHAPES
    ]
    geometric_mean = (
        math.prod(large_ratios) ** (1.0 / len(large_ratios))
        if len(large_ratios) == len(LARGE_SHAPES)
        else None
    )
    return {"rows": result_rows, "large_shape_geometric_mean": geometric_mean}


def write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "| M | N | K | sgBLAS GFLOP/s | cuBLAS GFLOP/s | Ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:  # type: ignore[index]
        lines.append(
            f"| {row['m']} | {row['n']} | {row['k']} | "
            f"{row['sgblas_gflops']:.1f} | {row['cublas_gflops']:.1f} | "
            f"{row['ratio']:.3f} |"
        )
    geometric_mean = summary["large_shape_geometric_mean"]
    if geometric_mean is not None:
        lines.extend(["", f"Large-shape geometric mean: **{geometric_mean:.4f}**"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_inventory(run_root: Path, manifest_path: Path) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact tree contains a symbolic link: {path}")
        if not path.is_file():
            continue
        if path == manifest_path or path.name == manifest_path.name + ".tmp":
            continue
        artifacts[path.relative_to(run_root).as_posix()] = artifact_file_record(
            path, run_root
        )
    return artifacts


def main() -> int:
    args = parse_args()
    container = container_provenance()
    campaign_started = time.monotonic()
    started_utc = utc_now()
    deadline = campaign_started + args.total_timeout_seconds
    source = args.source.resolve()
    build_root = (source / args.build_root).resolve()
    results_root = (source / args.results_root).resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest, source_files = source_snapshot(source)
    run_root = results_root / f"{timestamp}-a100-{digest[:12]}"
    run_root.mkdir(parents=True)
    campaign_build_root = build_root / digest / timestamp
    variants = {
        "wide": [
            "-DSGBLAS_EXPERIMENTAL_SM80_ASYNC=ON",
            "-DSGBLAS_EXPERIMENTAL_SM80_MEDIUM=OFF",
            "-DSGBLAS_EXPERIMENTAL_SM80_SMALL=OFF",
        ],
        "hybrid": [
            "-DSGBLAS_EXPERIMENTAL_SM80_ASYNC=ON",
            "-DSGBLAS_EXPERIMENTAL_SM80_MEDIUM=ON",
            "-DSGBLAS_SM80_MEDIUM_THREAD_ROWS=32",
            "-DSGBLAS_SM80_MEDIUM_MIN_WIDE_CTAS=128",
            "-DSGBLAS_SM80_MEDIUM_MAX_WIDE_CTAS=2147483647",
            "-DSGBLAS_SM80_MEDIUM_N_MAJOR_RASTER=OFF",
            "-DSGBLAS_SM80_MEDIUM_L2_PREFETCH_BYTES=0",
            "-DSGBLAS_SM80_MEDIUM_STAGES=2",
            "-DSGBLAS_EXPERIMENTAL_SM80_SMALL=ON",
            "-DSGBLAS_SM80_SMALL_TILE_COLUMNS=32",
            "-DSGBLAS_SM80_SMALL_THREAD_ROWS=32",
            "-DSGBLAS_SM80_SMALL_MAX_WIDE_CTAS=128",
            "-DSGBLAS_SM80_SMALL_SECOND_MIN_WIDE_CTAS=196",
            "-DSGBLAS_SM80_SMALL_SECOND_MAX_WIDE_CTAS=256",
            "-DSGBLAS_SM80_SMALL_SECOND_MAX_M=2048",
            "-DSGBLAS_SM80_SMALL_SECOND_MAX_N=4096",
            "-DSGBLAS_SM80_SMALL_MIN_BLOCKS_PER_SM=5",
            "-DSGBLAS_SM80_SMALL_MIN_K=128",
        ],
    }
    full_order = [
        ["wide", "hybrid"] if index % 2 == 0 else ["hybrid", "wide"]
        for index in range(args.runs)
    ]
    extra_order = [list(reversed(order)) for order in full_order]
    timing_order_full = balanced_timing_orders(args.runs)
    timing_order_8192 = {
        name: list(reversed(schedule)) for name, schedule in timing_order_full.items()
    }
    prime_order = {"wide": "sgblas-first", "hybrid": "cublas-first"}
    base_env = child_environment()
    benchmark_env = {**base_env, "NVIDIA_TF32_OVERRIDE": "0"}
    commands: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "schema_version": 1,
        "state": "incomplete",
        "started_utc": started_utc,
        "finished_utc": None,
        "duration_seconds": None,
        "failure": None,
        "container": container,
        "source": {
            "path": str(source),
            "sha256_before": digest,
            "sha256_after": None,
            "files": source_files,
            "git_before": None,
            "git_after": None,
        },
        "options": {
            "build_root": str(campaign_build_root),
            "results_root": str(results_root),
            "runs": args.runs,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "jobs": args.jobs,
            "seed": args.seed,
            "include_8192": args.include_8192,
            "repeats_8192": 50 if args.include_8192 else None,
            "sanitizers": args.sanitizers,
            "command_timeout_seconds": args.command_timeout_seconds,
            "total_timeout_seconds": args.total_timeout_seconds,
            "tf32_override": "0",
            "full_order": full_order,
            "order_8192": extra_order if args.include_8192 else None,
            "timing_order_prime": prime_order,
            "timing_order_full": timing_order_full,
            "timing_order_8192": timing_order_8192 if args.include_8192 else None,
        },
        "variants": variants,
        "child_environments": {"base": base_env, "benchmark": benchmark_env},
        "commands": commands,
        "probe": None,
        "telemetry": [],
        "binaries": {},
        "binary_copy_proofs": {},
        "build_evidence": {},
        "compiler_spill_reports": {},
        "artifacts": {},
        "manifest_excluded_from_artifact_hashes": True,
    }
    manifest_path = run_root / "manifest.json"
    write_json(manifest_path, manifest)

    def persist() -> None:
        write_json(manifest_path, manifest)

    runner = CommandRunner(
        deadline=deadline,
        command_timeout=args.command_timeout_seconds,
        artifact_root=run_root,
        environments={"base": base_env, "benchmark": benchmark_env},
        commands=commands,
        persist=persist,
    )

    try:
        if campaign_build_root.exists():
            raise RuntimeError(f"fresh build root already exists: {campaign_build_root}")

        nvidia_smi = find_tool("nvidia-smi")
        nvcc = find_tool("nvcc", "/usr/local/cuda/bin/nvcc")
        cmake = find_tool("cmake")
        git = find_tool("git")
        uname = find_tool("uname")
        manifest["tools"] = {
            "nvidia_smi": nvidia_smi,
            "nvcc": nvcc,
            "cmake": cmake,
            "git": git,
            "uname": uname,
        }
        persist()

        head_before = runner.capture(
            [git, "-C", str(source), "rev-parse", "--verify", "HEAD"], source
        )
        status_before = runner.capture(
            [git, "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
            source,
        )
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head_before) is None:
            raise RuntimeError("source must have a non-null Git HEAD")
        if status_before:
            raise RuntimeError("source Git worktree must be clean before the campaign")
        git_before = {
            "head": head_before,
            "status_porcelain": status_before,
            "available": True,
            "dirty": False,
        }
        manifest["source"]["git_before"] = git_before  # type: ignore[index]

        gpu_query = runner.capture(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            source,
        )
        gpu_identity = validate_gpu_query(gpu_query)
        nvidia_smi_q = redact_nvidia_smi_q(
            runner.capture([nvidia_smi, "-q"], source)
        )
        if re.search(r"\bGPU-[0-9A-Fa-f-]{16,}\b", nvidia_smi_q):
            raise RuntimeError("GPU UUID remained after nvidia-smi -q redaction")
        os_release_path = Path("/etc/os-release")
        if not os_release_path.is_file():
            raise RuntimeError("required host provenance file is missing: /etc/os-release")
        os_release = os_release_path.read_text(encoding="utf-8")
        initial_processes = parse_compute_processes(
            runner.capture(
                [
                    nvidia_smi,
                    "--query-compute-apps=" + ",".join(COMPUTE_PROCESS_FIELDS),
                    "--format=csv,noheader,nounits",
                ],
                source,
            )
        )
        probe = {
            "nvidia_smi": gpu_query,
            "gpu_identity": gpu_identity,
            "nvidia_smi_q": nvidia_smi_q,
            "nvidia_smi_q_identifiers_redacted": True,
            "active_compute_processes": initial_processes,
            "nvcc": runner.capture([nvcc, "--version"], source),
            "cmake": runner.capture([cmake, "--version"], source).splitlines()[0],
            "uname_a": runner.capture([uname, "-a"], source),
            "os_release": os_release,
            "os_release_sha256": hashlib.sha256(
                os_release.encode("utf-8")
            ).hexdigest(),
            "python": sys.version,
        }
        manifest["probe"] = probe
        persist()

        def capture_telemetry(
            *, phase: str, workload: str, variant: str, run_index: int
        ) -> None:
            started = utc_now()
            gpu = parse_gpu_telemetry(
                runner.capture(
                    [
                        nvidia_smi,
                        "--query-gpu=" + ",".join(TELEMETRY_FIELDS),
                        "--format=csv,noheader,nounits",
                    ],
                    source,
                )
            )
            processes = parse_compute_processes(
                runner.capture(
                    [
                        nvidia_smi,
                        "--query-compute-apps=" + ",".join(
                            COMPUTE_PROCESS_FIELDS
                        ),
                        "--format=csv,noheader,nounits",
                    ],
                    source,
                )
            )
            manifest["telemetry"].append(  # type: ignore[union-attr]
                {
                    "phase": phase,
                    "workload": workload,
                    "variant": variant,
                    "run_index": run_index,
                    "started_utc": started,
                    "finished_utc": utc_now(),
                    "gpu": gpu,
                    "active_compute_processes": processes,
                }
            )
            persist()

        benchmark_paths: dict[str, Path] = {}
        correctness_paths: dict[str, Path] = {}
        binary_original_records: dict[
            str, dict[str, dict[str, object]]
        ] = {}
        for name, definitions in variants.items():
            build = campaign_build_root / name
            variant_root = run_root / name
            configure = [
                cmake,
                "-S",
                str(source),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_ARCHITECTURES=80",
                "-DSGBLAS_ENABLE_CUDA=ON",
                "-DSGBLAS_BUILD_TESTS=ON",
                "-DSGBLAS_BUILD_BENCHMARKS=ON",
                *definitions,
            ]
            runner.logged(configure, source, variant_root / "configure.log")
            build_command = [cmake, "--build", str(build), "-j", str(args.jobs)]
            runner.logged(build_command, source, variant_root / "build.log")
            build_text = (variant_root / "build.log").read_text(encoding="utf-8")
            spills = [tuple(map(int, match)) for match in SPILL_PATTERN.findall(build_text)]
            if not spills:
                raise RuntimeError(f"{name} build emitted no compiler spill reports")
            if any(stores or loads for stores, loads in spills):
                raise RuntimeError(f"{name} generated register spills")
            manifest["compiler_spill_reports"][name] = [  # type: ignore[index]
                {"stores_bytes": stores, "loads_bytes": loads}
                for stores, loads in spills
            ]
            correctness = build / "sgblas_cuda_correctness"
            benchmark = build / "sgblas_benchmark"
            if not correctness.is_file() or not benchmark.is_file():
                raise RuntimeError(f"{name} build did not produce both required binaries")
            binary_original_records[name] = {
                "correctness": file_record(correctness),
                "benchmark": file_record(benchmark),
            }
            manifest["binary_copy_proofs"][name] = {  # type: ignore[index]
                kind: {
                    "original": record,
                    "copied": None,
                    "sha256_equal": None,
                    "size_equal": None,
                }
                for kind, record in binary_original_records[name].items()
            }
            persist()
            runner.logged([str(correctness)], source, variant_root / "correctness.log")
            correctness_paths[name] = correctness
            benchmark_paths[name] = benchmark

        for name, benchmark in benchmark_paths.items():
            prime_path = run_root / name / "prime.log"
            prime_command = [
                str(benchmark),
                "4096",
                "4096",
                "4096",
                "--warmups",
                "10",
                "--repeats",
                "20",
                "--seed",
                str(args.seed),
                "--order",
                prime_order[name],
                "--output",
                "jsonl",
            ]
            capture_telemetry(
                phase="before", workload="prime", variant=name, run_index=1
            )
            runner.logged(
                prime_command,
                source,
                prime_path,
                environment="benchmark",
            )
            validate_log(
                prime_path,
                ((4096, 4096, 4096),),
                expected_order=prime_order[name],
                expected_warmups=10,
                expected_repeats=20,
                expected_seed=args.seed,
                expected_argv=prime_command,
            )

        logs: dict[str, list[Path]] = {name: [] for name in variants}
        for run_index, order in enumerate(full_order, start=1):
            for name in order:
                path = run_root / name / f"full-{run_index:02d}.log"
                timed_order = timing_order_full[name][run_index - 1]
                benchmark_command = [
                    str(benchmark_paths[name]),
                    "--warmups",
                    str(args.warmups),
                    "--repeats",
                    str(args.repeats),
                    "--seed",
                    str(args.seed),
                    "--order",
                    timed_order,
                    "--output",
                    "jsonl",
                ]
                capture_telemetry(
                    phase="before",
                    workload="full",
                    variant=name,
                    run_index=run_index,
                )
                runner.logged(
                    benchmark_command,
                    source,
                    path,
                    environment="benchmark",
                )
                capture_telemetry(
                    phase="after",
                    workload="full",
                    variant=name,
                    run_index=run_index,
                )
                validate_log(
                    path,
                    DEFAULT_SHAPES,
                    expected_order=timed_order,
                    expected_warmups=args.warmups,
                    expected_repeats=args.repeats,
                    expected_seed=args.seed,
                    expected_argv=benchmark_command,
                )
                logs[name].append(path)

        if args.include_8192:
            expected_8192 = ((8192, 8192, 8192),)
            for run_index, order in enumerate(extra_order, start=1):
                for name in order:
                    path = run_root / name / f"8192-{run_index:02d}.log"
                    timed_order = timing_order_8192[name][run_index - 1]
                    benchmark_command = [
                        str(benchmark_paths[name]),
                        "8192",
                        "8192",
                        "8192",
                        "--warmups",
                        str(args.warmups),
                        "--repeats",
                        "50",
                        "--seed",
                        str(args.seed),
                        "--order",
                        timed_order,
                        "--output",
                        "jsonl",
                    ]
                    capture_telemetry(
                        phase="before",
                        workload="8192",
                        variant=name,
                        run_index=run_index,
                    )
                    runner.logged(
                        benchmark_command,
                        source,
                        path,
                        environment="benchmark",
                    )
                    capture_telemetry(
                        phase="after",
                        workload="8192",
                        variant=name,
                        run_index=run_index,
                    )
                    validate_log(
                        path,
                        expected_8192,
                        expected_order=timed_order,
                        expected_warmups=args.warmups,
                        expected_repeats=50,
                        expected_seed=args.seed,
                        expected_argv=benchmark_command,
                    )
                    logs[name].append(path)

        expected_telemetry = [
            ("before", "prime", name, 1) for name in variants
        ]
        for run_index, order in enumerate(full_order, start=1):
            for name in order:
                expected_telemetry.extend(
                    (
                        ("before", "full", name, run_index),
                        ("after", "full", name, run_index),
                    )
                )
        if args.include_8192:
            for run_index, order in enumerate(extra_order, start=1):
                for name in order:
                    expected_telemetry.extend(
                        (
                            ("before", "8192", name, run_index),
                            ("after", "8192", name, run_index),
                        )
                    )
        actual_telemetry = [
            (
                record["phase"],
                record["workload"],
                record["variant"],
                record["run_index"],
            )
            for record in manifest["telemetry"]  # type: ignore[union-attr]
        ]
        if actual_telemetry != expected_telemetry:
            raise RuntimeError("structured telemetry schedule is incomplete or reordered")

        if args.sanitizers:
            compute_sanitizer = find_tool(
                "compute-sanitizer", "/usr/local/cuda/bin/compute-sanitizer"
            )
            manifest["tools"]["compute_sanitizer"] = compute_sanitizer  # type: ignore[index]
            persist()
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                runner.logged(
                    [
                        compute_sanitizer,
                        "--tool",
                        tool,
                        "--error-exitcode=99",
                        str(correctness_paths["hybrid"]),
                    ],
                    source,
                    run_root / "hybrid" / f"sanitizer-{tool}.log",
                )

        expected_counts = {shape: args.runs for shape in DEFAULT_SHAPES}
        if args.include_8192:
            expected_counts[(8192, 8192, 8192)] = args.runs
        summaries = {
            name: summarize(paths, expected_counts) for name, paths in logs.items()
        }
        write_json(run_root / "summary.json", summaries)
        for name, summary in summaries.items():
            write_markdown(summary, run_root / f"summary-{name}.md")

        tested_binary_paths: dict[str, dict[str, Path]] = {}
        cmake_cache_paths: dict[str, tuple[Path, Path]] = {}
        for name in variants:
            runner.ensure_time(f"{name} tested-binary capture")
            tested_root = run_root / name / "tested-binaries"
            originals = {
                "correctness": correctness_paths[name],
                "benchmark": benchmark_paths[name],
            }
            destinations = {
                "correctness": tested_root / "sgblas_cuda_correctness",
                "benchmark": tested_root / "sgblas_benchmark",
            }
            proofs: dict[str, dict[str, object]] = {}
            for kind, original in originals.items():
                if file_record(original) != binary_original_records[name][kind]:
                    raise RuntimeError(
                        f"{name} {kind} binary changed before evidence capture"
                    )
                proofs[kind] = copy_verified_artifact(
                    original, destinations[kind], run_root
                )

            cmake_cache = campaign_build_root / name / "CMakeCache.txt"
            copied_cache = tested_root / "CMakeCache.txt"
            cache_proof = copy_verified_artifact(
                cmake_cache, copied_cache, run_root
            )
            configure_log = run_root / name / "configure.log"
            build_log = run_root / name / "build.log"
            manifest["binaries"][name] = {  # type: ignore[index]
                kind: proof["copied"] for kind, proof in proofs.items()
            }
            manifest["binary_copy_proofs"][name] = proofs  # type: ignore[index]
            manifest["build_evidence"][name] = {  # type: ignore[index]
                "cmake_cache": cache_proof,
                "configure_log": artifact_file_record(configure_log, run_root),
                "build_log": artifact_file_record(build_log, run_root),
            }
            tested_binary_paths[name] = destinations
            cmake_cache_paths[name] = (cmake_cache, copied_cache)
            persist()

        head_after = runner.capture(
            [git, "-C", str(source), "rev-parse", "--verify", "HEAD"], source
        )
        status_after = runner.capture(
            [git, "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
            source,
        )
        git_after = {
            "head": head_after,
            "status_porcelain": status_after,
            "available": True,
            "dirty": bool(status_after),
        }
        manifest["source"]["git_after"] = git_after  # type: ignore[index]
        digest_after, source_files_after = source_snapshot(source)
        manifest["source"]["sha256_after"] = digest_after  # type: ignore[index]
        if digest_after != digest or source_files_after != source_files:
            raise RuntimeError("source changed during the campaign")
        if head_after != head_before or status_after:
            raise RuntimeError("Git HEAD changed or worktree became dirty during the campaign")

        for name in variants:
            for kind, original in (
                ("correctness", correctness_paths[name]),
                ("benchmark", benchmark_paths[name]),
            ):
                if file_record(original) != binary_original_records[name][kind]:
                    raise RuntimeError(f"{name} {kind} binary changed during the campaign")
                proof = manifest["binary_copy_proofs"][name][kind]  # type: ignore[index]
                revalidate_artifact_copy(
                    original,
                    tested_binary_paths[name][kind],
                    run_root,
                    proof,
                )
                copied_record = artifact_file_record(
                    tested_binary_paths[name][kind], run_root
                )
                if copied_record != manifest["binaries"][name][kind]:  # type: ignore[index]
                    raise RuntimeError(f"{name} {kind} tested copy changed")

            cmake_cache, copied_cache = cmake_cache_paths[name]
            cache_proof = manifest["build_evidence"][name]["cmake_cache"]  # type: ignore[index]
            revalidate_artifact_copy(
                cmake_cache, copied_cache, run_root, cache_proof
            )
            for log_name, log_path in (
                ("configure_log", run_root / name / "configure.log"),
                ("build_log", run_root / name / "build.log"),
            ):
                if artifact_file_record(log_path, run_root) != manifest[
                    "build_evidence"
                ][name][log_name]:  # type: ignore[index]
                    raise RuntimeError(f"{name} {log_name} changed after capture")

        runner.ensure_time("artifact hashing")
        artifacts = artifact_inventory(run_root, manifest_path)
        manifest["artifacts"] = artifacts
        for name in variants:
            required_records = [
                *manifest["binaries"][name].values(),  # type: ignore[index]
                manifest["build_evidence"][name]["cmake_cache"]["copied"],  # type: ignore[index]
                manifest["build_evidence"][name]["configure_log"],  # type: ignore[index]
                manifest["build_evidence"][name]["build_log"],  # type: ignore[index]
            ]
            for record in required_records:
                relative = record["path"]
                if artifacts.get(relative) != record:
                    raise RuntimeError(
                        f"required evidence is absent or changed in inventory: {relative}"
                    )
        runner.ensure_time("campaign completion")
        manifest.update(
            {
                "state": "complete",
                "finished_utc": utc_now(),
                "duration_seconds": round(time.monotonic() - campaign_started, 6),
            }
        )
        persist()
        print(f"Tuning results: {run_root}")
        return 0
    except BaseException as error:
        manifest.update(
            {
                "state": "incomplete",
                "finished_utc": utc_now(),
                "duration_seconds": round(time.monotonic() - campaign_started, 6),
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        try:
            manifest["artifacts"] = artifact_inventory(run_root, manifest_path)
            persist()
        except Exception as manifest_error:
            print(f"could not finalize incomplete manifest: {manifest_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("tuning error: interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"tuning error: {error}", file=sys.stderr)
        raise SystemExit(1)
