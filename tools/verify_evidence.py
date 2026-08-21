#!/usr/bin/env python3
"""Fail-closed verification for an sgBLAS A100 tuning artifact bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, NamedTuple


SCHEMA_VERSION = 1
VARIANTS = ("wide", "hybrid")
DEFAULT_SHAPES = (
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 1024, 4096),
    (1024, 4096, 4096),
)
OPTIONAL_SHAPE = (8192, 8192, 8192)
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
EXPECTED_VARIANTS = {
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
SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
TIMING_ORDERS = ("sgblas-first", "cublas-first")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_SHA = re.compile(r"[0-9a-f]{40,64}\Z")
CORRECTNESS_PASS = re.compile(
    r"All (\d+) matrix-product cases and (\d+) quick-return/scale cases passed\."
)
RACECHECK_PASS = re.compile(
    r"RACECHECK SUMMARY:\s*0 hazards displayed \(0 errors, 0 warnings\)"
)
ERROR_SUMMARY = re.compile(r"ERROR SUMMARY:\s*(\d+) errors")
SPILL_PATTERN = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
MANIFEST_FIELDS = {
    "schema_version",
    "state",
    "started_utc",
    "finished_utc",
    "duration_seconds",
    "failure",
    "container",
    "source",
    "options",
    "variants",
    "child_environments",
    "commands",
    "probe",
    "telemetry",
    "binaries",
    "binary_copy_proofs",
    "build_evidence",
    "compiler_spill_reports",
    "artifacts",
    "manifest_excluded_from_artifact_hashes",
    "tools",
}
OPTION_FIELDS = {
    "build_root",
    "results_root",
    "runs",
    "warmups",
    "repeats",
    "jobs",
    "seed",
    "include_8192",
    "repeats_8192",
    "sanitizers",
    "command_timeout_seconds",
    "total_timeout_seconds",
    "tf32_override",
    "full_order",
    "order_8192",
    "timing_order_prime",
    "timing_order_full",
    "timing_order_8192",
}


class VerificationError(RuntimeError):
    """Raised when any evidence gate cannot be proven from the bundle."""


class DuplicateJsonKey(ValueError):
    pass


class BinaryBinding(NamedTuple):
    """Bind an acquisition-time binary path to its verified portable copy."""

    original_path: str
    copied_path: Path
    size_bytes: int
    sha256: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def expect_dict(value: object, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value  # type: ignore[return-value]


def expect_list(value: object, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value  # type: ignore[return-value]


def expect_string(value: object, label: str, *, nonempty: bool = True) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    if nonempty:
        require(bool(value), f"{label} must not be empty")
    return value


def expect_int(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer")
    return value  # type: ignore[return-value]


def expect_bool(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} must be a boolean")
    return value  # type: ignore[return-value]


def expect_number(value: object, label: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    if positive:
        require(result > 0.0, f"{label} must be positive")
    return result


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=strict_object,
            parse_constant=reject_json_constant,
        )
    except (json.JSONDecodeError, DuplicateJsonKey, ValueError) as error:
        raise VerificationError(f"invalid JSON in {label}: {error}") from error


def load_json(path: Path) -> Any:
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    return parse_json(contents, str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise VerificationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> None:
    require(not path.is_symlink(), f"{label} must not be a symbolic link: {path}")
    require(path.is_file(), f"{label} is missing or not a regular file: {path}")


def validate_sha256(value: object, label: str) -> str:
    digest = expect_string(value, label)
    require(HEX_SHA256.fullmatch(digest) is not None, f"{label} is not SHA-256")
    return digest


def validate_file_record(
    value: object,
    path: Path,
    label: str,
    *,
    record_has_path: bool = True,
    expected_record_path: str | None = None,
) -> dict[str, Any]:
    record = expect_dict(value, label)
    expected_keys = {"size_bytes", "sha256"}
    if record_has_path:
        expected_keys.add("path")
    require(set(record) == expected_keys, f"{label} has unexpected fields")
    require_regular_file(path, label)
    if record_has_path:
        recorded_path = expect_string(record.get("path"), f"{label}.path")
        require(expected_record_path is not None, f"{label} path policy is missing")
        require(
            recorded_path == expected_record_path,
            f"{label}.path does not match its portable location",
        )
    size = expect_int(record.get("size_bytes"), f"{label}.size_bytes")
    require(size >= 0, f"{label}.size_bytes must be nonnegative")
    require(path.stat().st_size == size, f"{label} size mismatch for {path}")
    digest = validate_sha256(record.get("sha256"), f"{label}.sha256")
    require(sha256_file(path) == digest, f"{label} checksum mismatch for {path}")
    return record


def portable_artifact_path(run_root: Path, value: object, label: str) -> tuple[str, Path]:
    text = expect_string(value, label)
    relative = PurePosixPath(text)
    require(
        text == relative.as_posix()
        and text not in (".", "")
        and not relative.is_absolute()
        and ".." not in relative.parts
        and "\\" not in text,
        f"{label} must be a canonical run-root-relative POSIX path",
    )
    path = (run_root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(run_root.resolve())
    except ValueError as error:
        raise VerificationError(f"{label} escapes the run root") from error
    return text, path


def validate_portable_file_record(
    value: object,
    run_root: Path,
    label: str,
    *,
    expected_relative: str | None = None,
) -> tuple[dict[str, Any], Path]:
    record = expect_dict(value, label)
    relative, path = portable_artifact_path(run_root, record.get("path"), f"{label}.path")
    if expected_relative is not None:
        require(relative == expected_relative, f"{label}.path mismatch")
    return (
        validate_file_record(
            record,
            path,
            label,
            expected_record_path=relative,
        ),
        path,
    )


def validate_detached_file_record(value: object, label: str) -> dict[str, Any]:
    """Validate metadata for an acquisition-time file that may no longer exist."""

    record = expect_dict(value, label)
    require(
        set(record) == {"path", "size_bytes", "sha256"},
        f"{label} has unexpected fields",
    )
    path = expect_string(record.get("path"), f"{label}.path")
    require(Path(path).is_absolute(), f"{label}.path must be absolute")
    size = expect_int(record.get("size_bytes"), f"{label}.size_bytes")
    require(size >= 0, f"{label}.size_bytes must be nonnegative")
    validate_sha256(record.get("sha256"), f"{label}.sha256")
    return record


def source_snapshot(source: Path) -> tuple[str, dict[str, dict[str, object]]]:
    digest = hashlib.sha256()
    files: list[Path] = []
    for entry in SOURCE_ROOTS:
        path = source / entry
        if path.is_symlink():
            raise VerificationError(f"source root must not be a symlink: {path}")
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_symlink():
                    raise VerificationError(
                        f"source snapshot contains a symlink: {candidate}"
                    )
                if candidate.is_file():
                    files.append(candidate)
        else:
            raise VerificationError(
                f"required source/provenance input is missing: {path}"
            )

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


def parse_utc(value: object, label: str) -> dt.datetime:
    text = expect_string(value, label)
    require(text.endswith("Z"), f"{label} must use a UTC Z suffix")
    try:
        parsed = dt.datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise VerificationError(f"{label} is not an ISO-8601 timestamp") from error
    require(parsed.tzinfo is not None, f"{label} must be timezone-aware")
    return parsed


def option_value(argv: list[str], name: str) -> str:
    positions = [index for index, item in enumerate(argv) if item == name]
    require(len(positions) == 1, f"command must contain exactly one {name}")
    position = positions[0]
    require(position + 1 < len(argv), f"command has no value after {name}")
    return argv[position + 1]


def validate_logged_header(path: Path, expected_argv: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"cannot read log {path}: {error}") from error
    command_lines = [line for line in lines if line.startswith("# command: ")]
    started_lines = [line for line in lines if line.startswith("# started_utc: ")]
    finished_lines = [line for line in lines if line.startswith("# finished_utc: ")]
    exit_lines = [line for line in lines if line.startswith("# exit_code: ")]
    require(len(command_lines) == 1, f"{path} must contain one command header")
    require(len(started_lines) == 1, f"{path} must contain one start timestamp")
    require(len(finished_lines) == 1, f"{path} must contain one finish timestamp")
    require(len(exit_lines) == 1, f"{path} must contain one exit-code footer")
    command = parse_json(command_lines[0][len("# command: ") :], f"{path} command")
    require(command == expected_argv, f"{path} command header does not match manifest")
    started = parse_utc(started_lines[0][len("# started_utc: ") :], f"{path} start")
    finished = parse_utc(
        finished_lines[0][len("# finished_utc: ") :], f"{path} finish"
    )
    require(finished >= started, f"{path} finishes before it starts")
    require(exit_lines[0] == "# exit_code: 0", f"{path} did not record exit code zero")


def command_argv(value: object, label: str) -> list[str]:
    raw = expect_list(value, label)
    require(raw, f"{label} must not be empty")
    require(all(isinstance(item, str) and item for item in raw), f"{label} is invalid")
    return raw  # type: ignore[return-value]


def validate_command_records(
    manifest: dict[str, Any], run_root: Path, artifact_paths: set[str]
) -> tuple[list[dict[str, Any]], dict[Path, dict[str, Any]]]:
    environments = expect_dict(manifest.get("child_environments"), "child_environments")
    commands = expect_list(manifest.get("commands"), "commands")
    require(commands, "commands must not be empty")
    by_log: dict[Path, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(commands):
        label = f"commands[{index}]"
        command = expect_dict(raw, label)
        argv = command_argv(command.get("argv"), f"{label}.argv")
        expect_string(command.get("cwd"), f"{label}.cwd")
        environment = expect_string(command.get("environment"), f"{label}.environment")
        require(environment in environments, f"{label} references an unknown environment")
        require(command.get("state") == "complete", f"{label} is not complete")
        require(command.get("exit_code") == 0, f"{label} did not exit successfully")
        require(command.get("error") in (None, ""), f"{label} records an error")
        started = parse_utc(command.get("started_utc"), f"{label}.started_utc")
        finished = parse_utc(command.get("finished_utc"), f"{label}.finished_utc")
        require(finished >= started, f"{label} finishes before it starts")
        duration = expect_number(command.get("duration_seconds"), f"{label}.duration_seconds")
        require(duration >= 0.0, f"{label}.duration_seconds must be nonnegative")
        if "output_sha256" in command:
            validate_sha256(command["output_sha256"], f"{label}.output_sha256")
        if "log_path" in command:
            log_text = expect_string(command.get("log_path"), f"{label}.log_path")
            relative, log_path = portable_artifact_path(
                run_root, log_text, f"{label}.log_path"
            )
            require(relative in artifact_paths, f"{label}.log_path is not inventoried")
            normalized_log_path = log_path.resolve()
            require(
                normalized_log_path not in by_log,
                f"duplicate command record for {log_path}",
            )
            validate_file_record(
                command.get("log"),
                log_path,
                f"{label}.log",
                expected_record_path=relative,
            )
            validate_logged_header(log_path, argv)
            by_log[normalized_log_path] = command
        normalized.append(command)
    return normalized, by_log


def validate_artifacts(
    manifest: dict[str, Any], run_root: Path, manifest_path: Path
) -> set[str]:
    require(
        manifest.get("manifest_excluded_from_artifact_hashes") is True,
        "manifest exclusion policy is missing",
    )
    require(
        not manifest_path.with_name(manifest_path.name + ".tmp").exists(),
        "temporary manifest remains in a complete bundle",
    )
    inventory = expect_dict(manifest.get("artifacts"), "artifacts")
    actual: dict[str, Path] = {}
    for candidate in run_root.rglob("*"):
        require(not candidate.is_symlink(), f"artifact bundle contains symlink: {candidate}")
        if candidate.is_file() and candidate != manifest_path:
            actual[candidate.relative_to(run_root).as_posix()] = candidate
    require(set(inventory) == set(actual), "artifact inventory file set mismatch")
    for relative, path in actual.items():
        relative_path = Path(relative)
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe artifact path: {relative}",
        )
        validate_file_record(
            inventory[relative],
            path,
            f"artifacts[{relative!r}]",
            expected_record_path=relative,
        )
    return set(actual)


def validate_source(
    manifest: dict[str, Any], source_override: Path | None
) -> tuple[Path, str, str]:
    source_data = expect_dict(manifest.get("source"), "source")
    require(
        set(source_data)
        == {
            "path",
            "sha256_before",
            "sha256_after",
            "files",
            "git_before",
            "git_after",
        },
        "source manifest has unexpected fields",
    )
    recorded_source = Path(expect_string(source_data.get("path"), "source.path"))
    require(recorded_source.is_absolute(), "source.path must be absolute")
    source = (source_override if source_override is not None else recorded_source).resolve()
    require(source.is_dir(), f"source directory is unavailable: {source}")
    before = validate_sha256(source_data.get("sha256_before"), "source.sha256_before")
    after = validate_sha256(source_data.get("sha256_after"), "source.sha256_after")
    require(before == after, "source digest changed during the campaign")

    recorded_files = expect_dict(source_data.get("files"), "source.files")
    for relative, record in recorded_files.items():
        require(isinstance(relative, str), "source.files keys must be strings")
        relative_path = Path(relative)
        require(
            relative and not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"unsafe source path in manifest: {relative}",
        )
        validate_file_record(
            record,
            source / relative_path,
            f"source.files[{relative!r}]",
            record_has_path=False,
        )
    digest, files = source_snapshot(source)
    require(files == recorded_files, "source file inventory does not match the source tree")
    require(digest == before, "source tree digest does not match the manifest")

    git_before = expect_dict(source_data.get("git_before"), "source.git_before")
    git_after = expect_dict(source_data.get("git_after"), "source.git_after")
    require(
        set(git_before) == {"head", "status_porcelain", "available", "dirty"},
        "source.git_before has unexpected fields",
    )
    require(
        set(git_after) == {"head", "status_porcelain", "available", "dirty"},
        "source.git_after has unexpected fields",
    )
    require(git_before == git_after, "Git state changed during the campaign")
    require(git_before.get("available") is True, "Git metadata is unavailable")
    require(git_before.get("dirty") is False, "evidence source tree was dirty")
    require(git_before.get("status_porcelain") == "", "evidence source tree was dirty")
    commit = expect_string(git_before.get("head"), "source.git_before.head")
    require(HEX_GIT_SHA.fullmatch(commit) is not None, "Git commit hash is invalid")
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "--verify", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        actual_status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.rstrip("\n")
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"cannot verify source Git state: {error}") from error
    require(actual_commit == commit, "current source commit does not match evidence")
    require(actual_status == "", "current source working tree is not clean")
    return source, before, commit


def validate_options(manifest: dict[str, Any]) -> dict[str, Any]:
    options = expect_dict(manifest.get("options"), "options")
    require(set(options) == OPTION_FIELDS, "options has unexpected or missing fields")
    runs = expect_int(options.get("runs"), "options.runs")
    require(runs >= 6 and runs % 2 == 0, "options.runs must be even and at least 6")
    for name in ("warmups", "repeats", "seed"):
        value = expect_int(options.get(name), f"options.{name}")
        require(value >= 0, f"options.{name} must be nonnegative")
    require(options["warmups"] >= 10, "options.warmups must be at least 10")
    require(options["repeats"] > 0, "options.repeats must be positive")
    require(options.get("tf32_override") == "0", "TF32 override must be 0")
    require(expect_bool(options.get("sanitizers"), "options.sanitizers"), "sanitizers required")
    jobs = expect_int(options.get("jobs"), "options.jobs")
    require(jobs > 0, "options.jobs must be positive")
    require(options["seed"] <= 0xFFFFFFFF, "options.seed is outside uint32 range")
    for name in ("command_timeout_seconds", "total_timeout_seconds"):
        value = expect_int(options.get(name), f"options.{name}")
        require(value > 0, f"options.{name} must be positive")
    for name in ("build_root", "results_root"):
        path = Path(expect_string(options.get(name), f"options.{name}"))
        require(path.is_absolute(), f"options.{name} must be absolute provenance")
    include_8192 = expect_bool(options.get("include_8192"), "options.include_8192")
    if include_8192:
        require(options.get("repeats_8192") == 50, "8192 repeat count must be 50")
    else:
        require(options.get("repeats_8192") is None, "unexpected 8192 repeat count")
    full_order = expect_list(options.get("full_order"), "options.full_order")
    require(len(full_order) == runs, "full-order schedule length mismatch")
    first_counts = {variant: 0 for variant in VARIANTS}
    for index, order_value in enumerate(full_order):
        order = expect_list(order_value, f"options.full_order[{index}]")
        require(len(order) == 2 and set(order) == set(VARIANTS), "invalid variant order")
        first_counts[order[0]] += 1
    require(len(set(first_counts.values())) == 1, "variant process order is unbalanced")
    expected_full_order = [
        ["wide", "hybrid"] if index % 2 == 0 else ["hybrid", "wide"]
        for index in range(runs)
    ]
    require(full_order == expected_full_order, "full-order schedule is not canonical")
    order_8192 = options.get("order_8192")
    if include_8192:
        extra = expect_list(order_8192, "options.order_8192")
        require(len(extra) == runs, "8192 order schedule length mismatch")
        for index, order_value in enumerate(extra):
            order = expect_list(order_value, f"options.order_8192[{index}]")
            require(len(order) == 2 and set(order) == set(VARIANTS), "invalid 8192 order")
        require(
            extra == [list(reversed(order)) for order in expected_full_order],
            "8192 process-order schedule is not canonical",
        )
    else:
        require(order_8192 is None, "unexpected 8192 order schedule")

    expected_prime = {"wide": "sgblas-first", "hybrid": "cublas-first"}
    prime_order = expect_dict(
        options.get("timing_order_prime"), "options.timing_order_prime"
    )
    require(prime_order == expected_prime, "prime timing-order schedule mismatch")
    expected_timing_full = {
        variant: [
            "sgblas-first" if (index + offset) % 2 == 0 else "cublas-first"
            for index in range(runs)
        ]
        for variant, offset in (("wide", 0), ("hybrid", 1))
    }
    timing_full = expect_dict(
        options.get("timing_order_full"), "options.timing_order_full"
    )
    require(
        timing_full == expected_timing_full,
        "full timing-order schedule mismatch",
    )
    timing_8192 = options.get("timing_order_8192")
    if include_8192:
        expected_timing_8192 = {
            variant: list(reversed(schedule))
            for variant, schedule in expected_timing_full.items()
        }
        require(
            expect_dict(timing_8192, "options.timing_order_8192")
            == expected_timing_8192,
            "8192 timing-order schedule mismatch",
        )
    else:
        require(timing_8192 is None, "unexpected 8192 timing-order schedule")

    variants = expect_dict(manifest.get("variants"), "variants")
    require(variants == EXPECTED_VARIANTS, "variant definitions do not match the evidence contract")
    environments = expect_dict(manifest.get("child_environments"), "child_environments")
    require(
        set(environments) == {"base", "benchmark"},
        "child environment set mismatch",
    )
    base = expect_dict(environments.get("base"), "child_environments.base")
    benchmark = expect_dict(environments.get("benchmark"), "child_environments.benchmark")
    require(
        all(isinstance(key, str) and isinstance(value, str) for key, value in base.items()),
        "base environment must contain only string pairs",
    )
    require(
        benchmark == {**base, "NVIDIA_TF32_OVERRIDE": "0"},
        "benchmark environment differs from base beyond TF32 override",
    )
    require(benchmark.get("NVIDIA_TF32_OVERRIDE") == "0", "benchmark environment permits TF32")
    for key, value in (
        ("LANG", "C"),
        ("LC_ALL", "C"),
        ("TZ", "UTC"),
        ("TMPDIR", "/tmp"),
    ):
        require(benchmark.get(key) == value, f"benchmark environment {key} mismatch")
    expect_string(benchmark.get("PATH"), "child_environments.benchmark.PATH")
    container = expect_dict(manifest.get("container"), "container")
    require(
        base.get("SGBLAS_CONTAINER_IMAGE") == container.get("image")
        and base.get("SGBLAS_CONTAINER_IMAGE_DIGEST")
        == container.get("image_digest"),
        "child environments do not preserve container provenance",
    )
    return options


def validate_container(manifest: dict[str, Any]) -> dict[str, str]:
    container = expect_dict(manifest.get("container"), "container")
    require(set(container) == {"image", "image_digest"}, "container provenance mismatch")
    output: dict[str, str] = {}
    for name in ("image", "image_digest"):
        value = expect_string(container.get(name), f"container.{name}")
        require(
            value == value.strip() and all(ord(character) >= 32 for character in value),
            f"container.{name} is not a valid provenance value",
        )
        output[name] = value
    return output


def validate_processes(value: object, label: str) -> list[dict[str, Any]]:
    processes = expect_list(value, label)
    seen: set[int] = set()
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(processes):
        item = f"{label}[{index}]"
        process = expect_dict(raw, item)
        require(
            set(process) == {"pid", "used_gpu_memory_mib"},
            f"{item} has unexpected fields",
        )
        pid = expect_int(process.get("pid"), f"{item}.pid")
        require(pid > 0 and pid not in seen, f"{item}.pid is invalid or duplicated")
        seen.add(pid)
        memory = process.get("used_gpu_memory_mib")
        if memory is not None:
            require(
                expect_int(memory, f"{item}.used_gpu_memory_mib") >= 0,
                f"{item}.used_gpu_memory_mib must be nonnegative",
            )
        output.append(process)
    return output


def parse_probe_identity(text: str) -> dict[str, object]:
    lines = [line for line in text.splitlines() if line.strip()]
    require(len(lines) == 1, "probe.nvidia_smi must identify exactly one GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    require(
        len(fields) == 4,
        "probe.nvidia_smi must contain name, driver, compute capability, and memory",
    )
    name, driver_version, compute_capability, memory_total = fields
    require(
        re.fullmatch(r"\d+(?:\.\d+){1,3}", driver_version) is not None,
        "probe NVIDIA driver version is invalid",
    )
    memory = re.fullmatch(r"(\d+)\s+MiB", memory_total)
    require(
        name == "NVIDIA A100-SXM4-80GB"
        and compute_capability == "8.0"
        and memory is not None
        and int(memory.group(1)) >= 80000,
        "probe is not exactly an A100-SXM4-80GB",
    )
    return {
        "name": name,
        "driver_version": driver_version,
        "compute_capability": compute_capability,
        "memory_total_mib": int(memory.group(1)),
    }


def validate_probe(manifest: dict[str, Any]) -> dict[str, object]:
    probe = expect_dict(manifest.get("probe"), "probe")
    require(
        set(probe)
        == {
            "nvidia_smi",
            "gpu_identity",
            "nvidia_smi_q",
            "nvidia_smi_q_identifiers_redacted",
            "active_compute_processes",
            "nvcc",
            "cmake",
            "uname_a",
            "os_release",
            "os_release_sha256",
            "python",
        },
        "probe has unexpected or missing fields",
    )
    nvidia_smi = expect_string(probe.get("nvidia_smi"), "probe.nvidia_smi")
    identity = parse_probe_identity(nvidia_smi)
    require(
        probe.get("gpu_identity") == identity,
        "probe.gpu_identity does not match nvidia-smi",
    )
    q_output = expect_string(probe.get("nvidia_smi_q"), "probe.nvidia_smi_q")
    require(
        probe.get("nvidia_smi_q_identifiers_redacted") is True,
        "nvidia-smi -q redaction marker is missing",
    )
    require(
        re.search(r"\bGPU-[0-9A-Fa-f-]{16,}\b", q_output) is None,
        "nvidia-smi -q contains an unredacted GPU UUID",
    )
    for match in re.finditer(
        r"(?im)^\s*(?:GPU UUID|Serial Number)\s*:\s*(.+?)\s*$", q_output
    ):
        require(match.group(1) == "<redacted>", "nvidia-smi -q identifier is not redacted")
    require("NVIDIA A100-SXM4-80GB" in q_output, "nvidia-smi -q lacks GPU model")
    validate_processes(probe.get("active_compute_processes"), "probe.active_compute_processes")
    require("Cuda compilation tools" in expect_string(probe.get("nvcc"), "probe.nvcc"), "nvcc probe incomplete")
    require(expect_string(probe.get("cmake"), "probe.cmake").startswith("cmake version "), "cmake probe incomplete")
    expect_string(probe.get("uname_a"), "probe.uname_a")
    os_release = expect_string(probe.get("os_release"), "probe.os_release")
    require(
        hashlib.sha256(os_release.encode("utf-8")).hexdigest()
        == validate_sha256(probe.get("os_release_sha256"), "probe.os_release_sha256"),
        "probe.os_release checksum mismatch",
    )
    expect_string(probe.get("python"), "probe.python")
    return identity


def validate_tools(manifest: dict[str, Any]) -> dict[str, str]:
    tools = expect_dict(manifest.get("tools"), "tools")
    expected = {
        "nvidia_smi",
        "nvcc",
        "cmake",
        "git",
        "uname",
        "compute_sanitizer",
    }
    require(set(tools) == expected, "tool manifest is incomplete")
    normalized: dict[str, str] = {}
    expected_names = {
        "nvidia_smi": "nvidia-smi",
        "nvcc": "nvcc",
        "cmake": "cmake",
        "git": "git",
        "uname": "uname",
        "compute_sanitizer": "compute-sanitizer",
    }
    for name, executable_name in expected_names.items():
        value = expect_string(tools.get(name), f"tools.{name}")
        require(Path(value).is_absolute(), f"tools.{name} must be absolute provenance")
        require(Path(value).name == executable_name, f"tools.{name} basename mismatch")
        normalized[name] = value
    return normalized


def validate_telemetry_gpu(
    value: object, label: str, identity: dict[str, object]
) -> None:
    gpu = expect_dict(value, label)
    expected_keys = {
        "reported_timestamp",
        "name",
        "driver_version",
        "performance_state",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "graphics_clock_mhz",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "clock_throttle_reasons_active",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "compute_mode",
        "mig_mode",
    }
    require(set(gpu) == expected_keys, f"{label} has unexpected or missing fields")
    expect_string(gpu.get("reported_timestamp"), f"{label}.reported_timestamp")
    require(gpu.get("name") == identity["name"], f"{label}.name mismatch")
    require(
        gpu.get("driver_version") == identity["driver_version"],
        f"{label}.driver_version mismatch",
    )
    require(
        re.fullmatch(
            r"P\d+", expect_string(gpu.get("performance_state"), f"{label}.performance_state")
        )
        is not None,
        f"{label}.performance_state is invalid",
    )
    for name in ("temperature_c", "graphics_clock_mhz", "sm_clock_mhz", "memory_clock_mhz"):
        require(expect_int(gpu.get(name), f"{label}.{name}") >= 0, f"{label}.{name} is negative")
    power_draw = expect_number(gpu.get("power_draw_w"), f"{label}.power_draw_w")
    power_limit = expect_number(gpu.get("power_limit_w"), f"{label}.power_limit_w")
    require(power_draw >= 0 and power_limit > 0, f"{label} power telemetry is invalid")
    for name in ("gpu_utilization_percent", "memory_utilization_percent"):
        utilization = expect_int(gpu.get(name), f"{label}.{name}")
        require(0 <= utilization <= 100, f"{label}.{name} is outside [0,100]")
    memory_used = expect_int(gpu.get("memory_used_mib"), f"{label}.memory_used_mib")
    memory_total = expect_int(gpu.get("memory_total_mib"), f"{label}.memory_total_mib")
    require(
        memory_total == identity["memory_total_mib"] and 0 <= memory_used <= memory_total,
        f"{label} memory telemetry mismatch",
    )
    for name in ("clock_throttle_reasons_active", "compute_mode", "mig_mode"):
        expect_string(gpu.get(name), f"{label}.{name}")


def validate_telemetry(
    manifest: dict[str, Any], options: dict[str, Any], identity: dict[str, object]
) -> None:
    records = expect_list(manifest.get("telemetry"), "telemetry")
    expected: list[tuple[str, str, str, int]] = [
        ("before", "prime", variant, 1) for variant in VARIANTS
    ]
    for run_index, process_order in enumerate(options["full_order"], start=1):
        for variant in process_order:
            expected.extend(
                (
                    ("before", "full", variant, run_index),
                    ("after", "full", variant, run_index),
                )
            )
    if options["include_8192"]:
        for run_index, process_order in enumerate(options["order_8192"], start=1):
            for variant in process_order:
                expected.extend(
                    (
                        ("before", "8192", variant, run_index),
                        ("after", "8192", variant, run_index),
                    )
                )
    require(len(records) == len(expected), "telemetry schedule length mismatch")
    for index, (raw, expected_key) in enumerate(zip(records, expected)):
        label = f"telemetry[{index}]"
        record = expect_dict(raw, label)
        require(
            set(record)
            == {
                "phase",
                "workload",
                "variant",
                "run_index",
                "started_utc",
                "finished_utc",
                "gpu",
                "active_compute_processes",
            },
            f"{label} has unexpected or missing fields",
        )
        actual_key = (
            record.get("phase"),
            record.get("workload"),
            record.get("variant"),
            record.get("run_index"),
        )
        require(actual_key == expected_key, f"{label} schedule mismatch")
        started = parse_utc(record.get("started_utc"), f"{label}.started_utc")
        finished = parse_utc(record.get("finished_utc"), f"{label}.finished_utc")
        require(finished >= started, f"{label} finishes before it starts")
        validate_telemetry_gpu(record.get("gpu"), f"{label}.gpu", identity)
        validate_processes(record.get("active_compute_processes"), f"{label}.active_compute_processes")


def validate_binaries(
    manifest: dict[str, Any], options: dict[str, Any], run_root: Path
) -> dict[str, dict[str, BinaryBinding]]:
    binary_data = expect_dict(manifest.get("binaries"), "binaries")
    require(set(binary_data) == set(VARIANTS), "binary variants mismatch")
    copy_proofs = expect_dict(manifest.get("binary_copy_proofs"), "binary_copy_proofs")
    require(set(copy_proofs) == set(VARIANTS), "binary copy proofs mismatch")
    build_root = Path(expect_string(options.get("build_root"), "options.build_root"))
    require(build_root.is_absolute(), "options.build_root must be absolute")
    result: dict[str, dict[str, BinaryBinding]] = {}
    filenames = {
        "benchmark": "sgblas_benchmark",
        "correctness": "sgblas_cuda_correctness",
    }
    for variant in VARIANTS:
        records = expect_dict(binary_data.get(variant), f"binaries.{variant}")
        require(set(records) == {"benchmark", "correctness"}, f"binaries.{variant} is incomplete")
        variant_proofs = expect_dict(
            copy_proofs.get(variant), f"binary_copy_proofs.{variant}"
        )
        require(
            set(variant_proofs) == {"benchmark", "correctness"},
            f"binary_copy_proofs.{variant} is incomplete",
        )
        result[variant] = {}
        for kind in ("benchmark", "correctness"):
            expected_relative = f"{variant}/tested-binaries/{filenames[kind]}"
            copied_record, copied_path = validate_portable_file_record(
                records.get(kind),
                run_root,
                f"binaries.{variant}.{kind}",
                expected_relative=expected_relative,
            )
            proof = expect_dict(
                variant_proofs.get(kind),
                f"binary_copy_proofs.{variant}.{kind}",
            )
            require(
                set(proof)
                == {"original", "copied", "sha256_equal", "size_equal"},
                f"binary_copy_proofs.{variant}.{kind} has unexpected fields",
            )
            original = validate_detached_file_record(
                proof.get("original"),
                f"binary_copy_proofs.{variant}.{kind}.original",
            )
            require(
                proof.get("copied") == copied_record,
                f"binary_copy_proofs.{variant}.{kind}.copied mismatch",
            )
            require(
                proof.get("sha256_equal") is True
                and proof.get("size_equal") is True,
                f"binary_copy_proofs.{variant}.{kind} does not prove equality",
            )
            require(
                original["sha256"] == copied_record["sha256"]
                and original["size_bytes"] == copied_record["size_bytes"],
                f"binary_copy_proofs.{variant}.{kind} metadata differs",
            )
            expected_original = (build_root / variant / filenames[kind]).resolve()
            original_path = expect_string(
                original.get("path"),
                f"binary_copy_proofs.{variant}.{kind}.original.path",
            )
            require(
                Path(original_path).resolve() == expected_original,
                f"binary_copy_proofs.{variant}.{kind}.original.path mismatch",
            )
            require(
                os.access(copied_path, os.X_OK),
                f"copied {variant} {kind} binary is not executable",
            )
            result[variant][kind] = BinaryBinding(
                original_path=original_path,
                copied_path=copied_path,
                size_bytes=expect_int(
                    copied_record.get("size_bytes"),
                    f"binaries.{variant}.{kind}.size_bytes",
                ),
                sha256=validate_sha256(
                    copied_record.get("sha256"),
                    f"binaries.{variant}.{kind}.sha256",
                ),
            )
    require(
        len({result[variant]["benchmark"].sha256 for variant in VARIANTS}) == 2,
        "wide and hybrid benchmark binaries are unexpectedly identical",
    )
    require(
        len({result[variant]["correctness"].sha256 for variant in VARIANTS}) == 2,
        "wide and hybrid correctness binaries are unexpectedly identical",
    )
    return result


def cmake_cache_values(path: Path, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise VerificationError(f"cannot read {label}: {error}") from error
    for line in lines:
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        name, remainder = line.split(":", 1)
        _, value = remainder.split("=", 1)
        require(name not in values, f"{label} duplicates CMake cache key {name}")
        values[name] = value
    return values


def validate_build_evidence(
    manifest: dict[str, Any], options: dict[str, Any], run_root: Path
) -> dict[str, dict[str, Path]]:
    evidence = expect_dict(manifest.get("build_evidence"), "build_evidence")
    require(set(evidence) == set(VARIANTS), "build evidence variants mismatch")
    build_root = Path(expect_string(options.get("build_root"), "options.build_root"))
    reports_by_variant = expect_dict(
        manifest.get("compiler_spill_reports"), "compiler_spill_reports"
    )
    require(
        set(reports_by_variant) == set(VARIANTS),
        "compiler spill reports are incomplete",
    )
    output: dict[str, dict[str, Path]] = {}
    base_cache_values = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": "80",
        "SGBLAS_ENABLE_CUDA": "ON",
        "SGBLAS_BUILD_TESTS": "ON",
        "SGBLAS_BUILD_BENCHMARKS": "ON",
    }
    spills = expect_dict(manifest.get("compiler_spill_reports"), "compiler_spill_reports")
    for variant in VARIANTS:
        variant_evidence = expect_dict(evidence.get(variant), f"build_evidence.{variant}")
        require(
            set(variant_evidence) == {"cmake_cache", "configure_log", "build_log"},
            f"build_evidence.{variant} is incomplete",
        )
        cache_proof = expect_dict(
            variant_evidence.get("cmake_cache"),
            f"build_evidence.{variant}.cmake_cache",
        )
        require(
            set(cache_proof)
            == {"original", "copied", "sha256_equal", "size_equal"},
            f"build_evidence.{variant}.cmake_cache has unexpected fields",
        )
        cache_original = validate_detached_file_record(
            cache_proof.get("original"),
            f"build_evidence.{variant}.cmake_cache.original",
        )
        cache_record, cache_path = validate_portable_file_record(
            cache_proof.get("copied"),
            run_root,
            f"build_evidence.{variant}.cmake_cache.copied",
            expected_relative=f"{variant}/tested-binaries/CMakeCache.txt",
        )
        require(
            cache_proof.get("sha256_equal") is True
            and cache_proof.get("size_equal") is True,
            f"build_evidence.{variant}.cmake_cache does not prove equality",
        )
        require(
            cache_original["sha256"] == cache_record["sha256"]
            and cache_original["size_bytes"] == cache_record["size_bytes"],
            f"build_evidence.{variant}.cmake_cache metadata differs",
        )
        require(
            Path(expect_string(cache_original.get("path"), "CMake cache original path")).resolve()
            == (build_root / variant / "CMakeCache.txt").resolve(),
            f"build_evidence.{variant}.cmake_cache original path mismatch",
        )
        configure_record, configure_path = validate_portable_file_record(
            variant_evidence.get("configure_log"),
            run_root,
            f"build_evidence.{variant}.configure_log",
            expected_relative=f"{variant}/configure.log",
        )
        build_record, build_path = validate_portable_file_record(
            variant_evidence.get("build_log"),
            run_root,
            f"build_evidence.{variant}.build_log",
            expected_relative=f"{variant}/build.log",
        )
        del configure_record, build_record
        cache = cmake_cache_values(cache_path, f"{variant} CMakeCache.txt")
        required_cache = dict(base_cache_values)
        for definition in EXPECTED_VARIANTS[variant]:
            require(definition.startswith("-D") and "=" in definition, "invalid expected variant")
            name, value = definition[2:].split("=", 1)
            required_cache[name] = value
        for name, value in required_cache.items():
            require(
                cache.get(name) == value,
                f"{variant} CMake cache does not prove {name}={value}",
            )

        reports = expect_list(spills[variant], f"compiler_spill_reports.{variant}")
        require(reports, f"compiler_spill_reports.{variant} is empty")
        for index, raw in enumerate(reports):
            report = expect_dict(raw, f"compiler_spill_reports.{variant}[{index}]")
            require(
                report == {"stores_bytes": 0, "loads_bytes": 0},
                f"{variant} compiler report contains spills or unknown fields",
            )
        raw_spills = [
            {"stores_bytes": int(stores), "loads_bytes": int(loads)}
            for stores, loads in SPILL_PATTERN.findall(
                build_path.read_text(encoding="utf-8")
            )
        ]
        require(raw_spills == reports, f"{variant} raw build spill reports mismatch")
        output[variant] = {
            "cache": cache_path,
            "configure_log": configure_path,
            "build_log": build_path,
        }
    return output


def command_for_log(
    by_log: dict[Path, dict[str, Any]], path: Path, label: str
) -> dict[str, Any]:
    command = by_log.get(path.resolve())
    require(command is not None, f"no command record for {label}: {path}")
    return command  # type: ignore[return-value]


def correctness_marker(text: str, label: str) -> tuple[int, int]:
    matches = CORRECTNESS_PASS.findall(text)
    require(len(matches) == 1, f"{label} must contain one correctness-pass marker")
    matrix_cases, quick_cases = (int(value) for value in matches[0])
    require(matrix_cases >= 10, f"{label} covers too few matrix-product cases")
    require(quick_cases >= 13, f"{label} covers too few quick-return/scale cases")
    require(re.search(r"\bFAIL(?:ED)?\b", text, re.IGNORECASE) is None, f"{label} contains a failure marker")
    return matrix_cases, quick_cases


def validate_correctness(
    run_root: Path,
    by_log: dict[Path, dict[str, Any]],
    binaries: dict[str, dict[str, BinaryBinding]],
) -> None:
    for variant in VARIANTS:
        path = run_root / variant / "correctness.log"
        require_regular_file(path, f"{variant} correctness log")
        command = command_for_log(by_log, path, f"{variant} correctness")
        argv = command_argv(command.get("argv"), f"{variant} correctness argv")
        require(
            argv == [binaries[variant]["correctness"].original_path],
            f"{variant} correctness command mismatch",
        )
        require(command.get("environment") == "base", f"{variant} correctness environment mismatch")
        text = path.read_text(encoding="utf-8")
        correctness_marker(text, f"{variant} correctness log")


def json_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        value = expect_dict(parse_json(stripped, f"{path}:{line_number}"), f"{path}:{line_number}")
        record_type = value.get("record_type")
        if record_type == "metadata":
            metadata.append(value)
        elif record_type == "result":
            results.append(value)
        else:
            raise VerificationError(f"unknown JSON record type in {path}:{line_number}")
    require(len(metadata) == 1, f"{path} must contain one metadata record")
    require(results, f"{path} contains no result records")
    return metadata[0], results


def validate_benchmark_contract(
    metadata: dict[str, Any],
    argv: list[str],
    expected_order: str,
    warmups: int,
    repeats: int,
    seed: int,
    label: str,
) -> tuple[Any, ...]:
    require(metadata.get("record_type") == "metadata", f"{label} metadata type mismatch")
    require(metadata.get("schema_version") == 1, f"{label} metadata schema mismatch")
    require(metadata.get("argv") == argv, f"{label} metadata argv mismatch")
    benchmark = expect_dict(metadata.get("benchmark"), f"{label}.benchmark")
    recorded_warmups = expect_int(
        benchmark.get("warmups_per_implementation"),
        f"{label}.benchmark.warmups_per_implementation",
    )
    require(
        recorded_warmups >= 10,
        f"{label} benchmark metadata records fewer than 10 warmups",
    )
    expected = {
        "timed_order": expected_order,
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
    }
    require(benchmark == expected, f"{label} benchmark contract mismatch")
    cuda = expect_dict(metadata.get("cuda"), f"{label}.cuda")
    cublas = expect_dict(metadata.get("cublas"), f"{label}.cublas")
    device = expect_dict(metadata.get("device"), f"{label}.device")
    require(
        set(device)
        == {
            "ordinal",
            "name",
            "pci_bus_id",
            "compute_capability_major",
            "compute_capability_minor",
            "multiprocessor_count",
            "total_global_memory_bytes",
            "l2_cache_bytes",
            "shared_memory_per_block_bytes",
            "shared_memory_per_block_optin_bytes",
            "shared_memory_per_multiprocessor_bytes",
            "registers_per_multiprocessor",
            "warp_size",
            "max_threads_per_multiprocessor",
            "reported_core_clock_khz",
            "reported_memory_clock_khz",
            "memory_bus_width_bits",
        },
        f"{label}.device contains unexpected fields or identifiers",
    )
    require(device.get("compute_capability_major") == 8, f"{label} is not SM80")
    require(device.get("compute_capability_minor") == 0, f"{label} is not SM80")
    require(
        expect_string(device.get("name"), f"{label}.device.name")
        == "NVIDIA A100-SXM4-80GB",
        f"{label} is not exactly an A100-SXM4-80GB",
    )
    require(expect_int(device.get("total_global_memory_bytes"), f"{label}.device.memory") >= 80_000_000_000, f"{label} is not an 80 GB device")
    for name in (
        "driver_version_raw",
        "runtime_version_raw",
    ):
        require(expect_int(cuda.get(name), f"{label}.cuda.{name}") > 0, f"{label} CUDA version invalid")
    require(expect_int(cublas.get("version_raw"), f"{label}.cublas.version_raw") > 0, f"{label} cuBLAS version invalid")
    return (cuda, cublas, device)


def validate_result_records(
    records: list[dict[str, Any]],
    expected_shapes: tuple[tuple[int, int, int], ...],
    order: str,
    warmups: int,
    repeats: int,
    label: str,
) -> dict[tuple[int, int, int], dict[str, float]]:
    require(len(records) == len(expected_shapes), f"{label} result count mismatch")
    output: dict[tuple[int, int, int], dict[str, float]] = {}
    actual_shapes: list[tuple[int, int, int]] = []
    for index, record in enumerate(records):
        item = f"{label}.results[{index}]"
        require(record.get("record_type") == "result", f"{item} type mismatch")
        require(record.get("schema_version") == 1, f"{item} schema mismatch")
        require(record.get("timed_order") == order, f"{item} timing order mismatch")
        require(record.get("warmups_per_implementation") == warmups, f"{item} warmup mismatch")
        require(record.get("timed_repeats_per_implementation") == repeats, f"{item} repeat mismatch")
        shape = tuple(expect_int(record.get(key), f"{item}.{key}") for key in ("m", "n", "k"))
        actual_shapes.append(shape)  # type: ignore[arg-type]
        values: dict[str, float] = {}
        for key in (
            "sgblas_total_ms",
            "sgblas_mean_ms",
            "sgblas_gflops",
            "cublas_total_ms",
            "cublas_mean_ms",
            "cublas_gflops",
            "ratio",
        ):
            values[key] = expect_number(record.get(key), f"{item}.{key}", positive=True)
        require(
            math.isclose(
                values["sgblas_total_ms"] / repeats,
                values["sgblas_mean_ms"],
                rel_tol=2e-5,
            ),
            f"{item} sgBLAS total/mean mismatch",
        )
        require(
            math.isclose(
                values["cublas_total_ms"] / repeats,
                values["cublas_mean_ms"],
                rel_tol=2e-5,
            ),
            f"{item} cuBLAS total/mean mismatch",
        )
        operation_count = 2.0 * shape[0] * shape[1] * shape[2]
        require(
            math.isclose(
                operation_count / (values["sgblas_mean_ms"] * 1.0e6),
                values["sgblas_gflops"],
                rel_tol=2e-5,
            ),
            f"{item} sgBLAS GFLOP/s mismatch",
        )
        require(
            math.isclose(
                operation_count / (values["cublas_mean_ms"] * 1.0e6),
                values["cublas_gflops"],
                rel_tol=2e-5,
            ),
            f"{item} cuBLAS GFLOP/s mismatch",
        )
        require(
            math.isclose(
                values["sgblas_gflops"] / values["cublas_gflops"],
                values["ratio"],
                rel_tol=2e-9,
            ),
            f"{item} ratio mismatch",
        )
        output[shape] = values  # type: ignore[index]
    require(tuple(actual_shapes) == expected_shapes, f"{label} shape corpus mismatch")
    return output


def validate_benchmark_log(
    path: Path,
    command: dict[str, Any],
    benchmark_binary: BinaryBinding,
    expected_shapes: tuple[tuple[int, int, int], ...],
    warmups: int,
    repeats: int,
    seed: int,
    label: str,
) -> tuple[str, tuple[Any, ...], dict[tuple[int, int, int], dict[str, float]]]:
    argv = command_argv(command.get("argv"), f"{label}.argv")
    require(
        argv[0] == benchmark_binary.original_path,
        f"{label} uses the wrong acquisition-time binary",
    )
    require(command.get("environment") == "benchmark", f"{label} uses the wrong environment")
    order = option_value(argv, "--order")
    require(order in TIMING_ORDERS, f"{label} timing order is invalid")
    require(option_value(argv, "--warmups") == str(warmups), f"{label} warmup argv mismatch")
    require(option_value(argv, "--repeats") == str(repeats), f"{label} repeat argv mismatch")
    require(option_value(argv, "--seed") == str(seed), f"{label} seed argv mismatch")
    require(option_value(argv, "--output") == "jsonl", f"{label} output mode mismatch")
    metadata, records = json_records(path)
    machine = validate_benchmark_contract(
        metadata, argv, order, warmups, repeats, seed, label
    )
    values = validate_result_records(
        records, expected_shapes, order, warmups, repeats, label
    )
    return order, machine, values


def validate_benchmarks(
    manifest: dict[str, Any],
    run_root: Path,
    by_log: dict[Path, dict[str, Any]],
    commands: list[dict[str, Any]],
    options: dict[str, Any],
    binaries: dict[str, dict[str, BinaryBinding]],
) -> tuple[dict[str, list[dict[tuple[int, int, int], dict[str, float]]]], tuple[tuple[int, int, int], ...]]:
    runs = options["runs"]
    warmups = options["warmups"]
    repeats = options["repeats"]
    seed = options["seed"]
    include_8192 = options["include_8192"]
    all_values: dict[str, list[dict[tuple[int, int, int], dict[str, float]]]] = {
        variant: [] for variant in VARIANTS
    }
    full_commands: dict[tuple[int, str], dict[str, Any]] = {}
    full_orders: dict[str, list[str]] = {variant: [] for variant in VARIANTS}
    full_hashes: set[str] = set()
    started_times: set[str] = set()
    reference_machine: tuple[Any, ...] | None = None

    for variant in VARIANTS:
        path = run_root / variant / "prime.log"
        require_regular_file(path, f"{variant} prime log")
        command = command_for_log(by_log, path, f"{variant} prime run")
        expected_order = options["timing_order_prime"][variant]
        order, machine, _ = validate_benchmark_log(
            path,
            command,
            binaries[variant]["benchmark"],
            ((4096, 4096, 4096),),
            10,
            20,
            seed,
            f"{variant} prime run",
        )
        require(order == expected_order, f"{variant} prime timing order mismatch")
        argv = command_argv(command.get("argv"), f"{variant} prime argv")
        expected_argv = [
            binaries[variant]["benchmark"].original_path,
            "4096",
            "4096",
            "4096",
            "--warmups",
            "10",
            "--repeats",
            "20",
            "--seed",
            str(seed),
            "--order",
            expected_order,
            "--output",
            "jsonl",
        ]
        require(argv == expected_argv, f"{variant} prime command mismatch")
        reference_machine = machine if reference_machine is None else reference_machine
        require(machine == reference_machine, "prime machine metadata changed")

    for variant in VARIANTS:
        expected_names = {f"full-{index:02d}.log" for index in range(1, runs + 1)}
        actual_names = {path.name for path in (run_root / variant).glob("full-*.log")}
        require(actual_names == expected_names, f"{variant} independent-log set mismatch")
        for index in range(1, runs + 1):
            path = run_root / variant / f"full-{index:02d}.log"
            command = command_for_log(by_log, path, f"{variant} full run {index}")
            order, machine, values = validate_benchmark_log(
                path,
                command,
                binaries[variant]["benchmark"],
                DEFAULT_SHAPES,
                warmups,
                repeats,
                seed,
                f"{variant} full run {index}",
            )
            full_orders[variant].append(order)
            require(
                order == options["timing_order_full"][variant][index - 1],
                f"{variant} full run {index} differs from timing schedule",
            )
            argv = command_argv(
                command.get("argv"), f"{variant} full run {index} argv"
            )
            expected_argv = [
                binaries[variant]["benchmark"].original_path,
                "--warmups",
                str(warmups),
                "--repeats",
                str(repeats),
                "--seed",
                str(seed),
                "--order",
                order,
                "--output",
                "jsonl",
            ]
            require(argv == expected_argv, f"{variant} full run {index} command mismatch")
            full_commands[(index, variant)] = command
            all_values[variant].append(values)
            reference_machine = machine if reference_machine is None else reference_machine
            require(machine == reference_machine, "benchmark machine metadata changed between logs")
            digest = sha256_file(path)
            require(digest not in full_hashes, "independent benchmark logs are byte-identical")
            full_hashes.add(digest)
            started = expect_string(command.get("started_utc"), "benchmark started_utc")
            require(started not in started_times, "independent benchmark commands share a start timestamp")
            started_times.add(started)

    for variant, orders in full_orders.items():
        counts = {order: orders.count(order) for order in TIMING_ORDERS}
        require(
            counts["sgblas-first"] == counts["cublas-first"] == runs // 2,
            f"{variant} library timing order is unbalanced",
        )

    positions = {id(command): index for index, command in enumerate(commands)}
    for index, raw_order in enumerate(options["full_order"], start=1):
        actual = sorted(
            VARIANTS,
            key=lambda variant: positions[id(full_commands[(index, variant)])],
        )
        require(actual == raw_order, f"full run {index} variant process order mismatch")

    expected_shapes = DEFAULT_SHAPES
    if include_8192:
        expected_shapes = DEFAULT_SHAPES + (OPTIONAL_SHAPE,)
        extra_orders: dict[str, list[str]] = {variant: [] for variant in VARIANTS}
        extra_commands: dict[tuple[int, str], dict[str, Any]] = {}
        for variant in VARIANTS:
            expected_names = {f"8192-{index:02d}.log" for index in range(1, runs + 1)}
            actual_names = {path.name for path in (run_root / variant).glob("8192-*.log")}
            require(actual_names == expected_names, f"{variant} 8192 log set mismatch")
            for index in range(1, runs + 1):
                path = run_root / variant / f"8192-{index:02d}.log"
                command = command_for_log(by_log, path, f"{variant} 8192 run {index}")
                order, machine, values = validate_benchmark_log(
                    path,
                    command,
                    binaries[variant]["benchmark"],
                    (OPTIONAL_SHAPE,),
                    warmups,
                    50,
                    seed,
                    f"{variant} 8192 run {index}",
                )
                extra_orders[variant].append(order)
                require(
                    order == options["timing_order_8192"][variant][index - 1],
                    f"{variant} 8192 run {index} differs from timing schedule",
                )
                argv = command_argv(
                    command.get("argv"), f"{variant} 8192 run {index} argv"
                )
                expected_argv = [
                    binaries[variant]["benchmark"].original_path,
                    "8192",
                    "8192",
                    "8192",
                    "--warmups",
                    str(warmups),
                    "--repeats",
                    "50",
                    "--seed",
                    str(seed),
                    "--order",
                    order,
                    "--output",
                    "jsonl",
                ]
                require(
                    argv == expected_argv,
                    f"{variant} 8192 run {index} command mismatch",
                )
                extra_commands[(index, variant)] = command
                all_values[variant][index - 1].update(values)
                require(machine == reference_machine, "8192 machine metadata changed")
        for variant, orders in extra_orders.items():
            require(
                orders.count("sgblas-first") == orders.count("cublas-first") == runs // 2,
                f"{variant} 8192 library timing order is unbalanced",
            )
        for index, raw_order in enumerate(options["order_8192"], start=1):
            actual = sorted(
                VARIANTS,
                key=lambda variant: positions[id(extra_commands[(index, variant)])],
            )
            require(actual == raw_order, f"8192 run {index} variant process order mismatch")
    else:
        for variant in VARIANTS:
            require(not list((run_root / variant).glob("8192-*.log")), f"unexpected {variant} 8192 logs")

    return all_values, expected_shapes


def validate_sanitizers(
    run_root: Path,
    by_log: dict[Path, dict[str, Any]],
    binaries: dict[str, dict[str, BinaryBinding]],
    tools: dict[str, str],
) -> None:
    for tool in SANITIZER_TOOLS:
        path = run_root / "hybrid" / f"sanitizer-{tool}.log"
        require_regular_file(path, f"{tool} sanitizer log")
        command = command_for_log(by_log, path, f"{tool} sanitizer")
        argv = command_argv(command.get("argv"), f"{tool} sanitizer argv")
        require(
            argv[0] == tools["compute_sanitizer"],
            f"{tool} sanitizer executable mismatch",
        )
        require(
            argv[1:]
            == [
                "--tool",
                tool,
                "--error-exitcode=99",
                binaries["hybrid"]["correctness"].original_path,
            ],
            f"{tool} sanitizer command mismatch",
        )
        require(command.get("environment") == "base", f"{tool} sanitizer environment mismatch")
        text = path.read_text(encoding="utf-8")
        require("COMPUTE-SANITIZER" in text, f"{tool} log lacks Compute Sanitizer marker")
        correctness_marker(text, f"{tool} sanitizer log")
        require("========= ERROR:" not in text, f"{tool} sanitizer reported an error")
        require("========= WARNING:" not in text, f"{tool} sanitizer reported a warning")
        if tool == "racecheck":
            require(RACECHECK_PASS.search(text) is not None, "racecheck clean summary missing")
        else:
            summaries = [int(value) for value in ERROR_SUMMARY.findall(text)]
            require(summaries == [0], f"{tool} must contain one zero-error summary")


def validate_summary(
    run_root: Path,
    values: dict[str, list[dict[tuple[int, int, int], dict[str, float]]]],
    expected_shapes: tuple[tuple[int, int, int], ...],
) -> None:
    summary_path = run_root / "summary.json"
    summary = expect_dict(load_json(summary_path), "summary.json")
    require(set(summary) == set(VARIANTS), "summary variants mismatch")
    for variant in VARIANTS:
        data = expect_dict(summary[variant], f"summary.{variant}")
        rows = expect_list(data.get("rows"), f"summary.{variant}.rows")
        require(len(rows) == len(expected_shapes), f"summary.{variant} row count mismatch")
        by_shape: dict[tuple[int, int, int], dict[str, Any]] = {}
        for index, raw in enumerate(rows):
            row = expect_dict(raw, f"summary.{variant}.rows[{index}]")
            shape = tuple(expect_int(row.get(key), f"summary.{variant}.{key}") for key in ("m", "n", "k"))
            require(shape not in by_shape, f"summary.{variant} duplicates {shape}")
            by_shape[shape] = row  # type: ignore[index]
        require(set(by_shape) == set(expected_shapes), f"summary.{variant} shapes mismatch")
        for shape in expected_shapes:
            row = by_shape[shape]
            samples = [run[shape] for run in values[variant]]
            require(row.get("runs") == len(samples), f"summary.{variant} run count mismatch for {shape}")
            expected = {
                "sgblas_ms": statistics.median(item["sgblas_mean_ms"] for item in samples),
                "sgblas_gflops": statistics.median(item["sgblas_gflops"] for item in samples),
                "sgblas_gflops_min": min(item["sgblas_gflops"] for item in samples),
                "sgblas_gflops_max": max(item["sgblas_gflops"] for item in samples),
                "cublas_ms": statistics.median(item["cublas_mean_ms"] for item in samples),
                "cublas_gflops": statistics.median(item["cublas_gflops"] for item in samples),
                "ratio": statistics.median(item["ratio"] for item in samples),
            }
            for key, expected_value in expected.items():
                actual = expect_number(row.get(key), f"summary.{variant}.{shape}.{key}")
                require(
                    math.isclose(actual, expected_value, rel_tol=2e-9, abs_tol=1e-12),
                    f"summary.{variant} {key} mismatch for {shape}",
                )
        require_regular_file(run_root / f"summary-{variant}.md", f"{variant} markdown summary")


def verify_evidence(run_root: Path, source_override: Path | None = None) -> dict[str, object]:
    run_root = run_root.resolve()
    require(run_root.is_dir(), f"artifact directory does not exist: {run_root}")
    manifest_path = run_root / "manifest.json"
    require_regular_file(manifest_path, "manifest")
    manifest = expect_dict(load_json(manifest_path), "manifest")
    require(set(manifest) == MANIFEST_FIELDS, "manifest has unexpected or missing fields")
    require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema mismatch")
    require(manifest.get("state") == "complete", "manifest state is not complete")
    require(manifest.get("failure") is None, "manifest records a campaign failure")
    started = parse_utc(manifest.get("started_utc"), "started_utc")
    finished = parse_utc(manifest.get("finished_utc"), "finished_utc")
    require(finished > started, "campaign finish must follow its start")
    duration = expect_number(manifest.get("duration_seconds"), "duration_seconds")
    require(duration > 0.0, "campaign duration must be positive")
    require(
        abs((finished - started).total_seconds() - duration) <= 2.0,
        "campaign duration disagrees with UTC timestamps",
    )

    artifact_paths = validate_artifacts(manifest, run_root, manifest_path)
    source, source_digest, commit = validate_source(manifest, source_override)
    validate_container(manifest)
    options = validate_options(manifest)
    identity = validate_probe(manifest)
    tools = validate_tools(manifest)
    validate_telemetry(manifest, options, identity)
    binaries = validate_binaries(manifest, options, run_root)
    validate_build_evidence(manifest, options, run_root)
    commands, by_log = validate_command_records(manifest, run_root, artifact_paths)
    validate_correctness(run_root, by_log, binaries)
    values, shapes = validate_benchmarks(
        manifest, run_root, by_log, commands, options, binaries
    )
    validate_sanitizers(run_root, by_log, binaries, tools)
    validate_summary(run_root, values, shapes)
    return {
        "run_root": str(run_root),
        "source": str(source),
        "source_sha256": source_digest,
        "git_commit": commit,
        "variants": list(VARIANTS),
        "shapes": [list(shape) for shape in shapes],
        "independent_logs_per_variant": options["runs"],
        "sanitizers": list(SANITIZER_TOOLS),
        "artifact_count": len(artifact_paths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify an sgBLAS A100 tuning artifact directory fail-closed."
    )
    parser.add_argument("run_root", type=Path, help="runner artifact directory")
    parser.add_argument(
        "--source",
        type=Path,
        help="source checkout override (defaults to manifest source.path)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = verify_evidence(args.run_root, args.source)
    except VerificationError as error:
        print(f"evidence verification failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            "Evidence verification PASSED: "
            f"{report['independent_logs_per_variant']} balanced logs per variant, "
            f"{len(report['shapes'])} shapes, four clean sanitizers, "
            f"{report['artifact_count']} checksummed artifacts"
        )
        print(f"Git commit: {report['git_commit']}")
        print(f"Source SHA-256: {report['source_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
