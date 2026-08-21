#!/usr/bin/env python3
"""Build and verify an sgBLAS release bundle bound to GPU evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable


TOOL_VERSION = 1
RELEASE_SCHEMA_VERSION = 1
PROJECT = "sgblas"
TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_LINE_RE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
SENSITIVE_PATTERNS = (
    ("NVIDIA GPU UUID", re.compile(rb"\bGPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained token", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
)


class ReleaseError(RuntimeError):
    """A release gate failed."""


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
    check: bool = True,
    stdout: Any = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if stdout is not None:
        capture = False
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=stdout is None,
        stdout=stdout if stdout is not None else (subprocess.PIPE if capture else None),
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        suffix = f": {detail}" if detail else ""
        raise ReleaseError(f"command failed ({' '.join(args)}){suffix}")
    return result


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = run(["git", "-C", str(repo), *args], check=check)
    return (result.stdout or "").strip()


def repository_root() -> Path:
    candidate = Path(__file__).resolve().parent.parent
    root = git(candidate, "rev-parse", "--show-toplevel")
    return Path(root).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{path} must contain a JSON object")
    return value


def ensure_clean(repo: Path) -> None:
    status_text = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status_text:
        preview = "\n".join(status_text.splitlines()[:12])
        raise ReleaseError(f"working tree is not clean:\n{preview}")


def resolve_commit(repo: Path, ref: str) -> str:
    commit = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseError(f"unexpected commit object name for {ref}: {commit}")
    return commit


def resolve_release_tag(repo: Path, tag: str, allow_lightweight: bool) -> tuple[str, bool]:
    if not TAG_RE.fullmatch(tag):
        raise ReleaseError(f"release tag must be SemVer-shaped (for example v0.1.0), got {tag!r}")
    ref = f"refs/tags/{tag}"
    if run(["git", "-C", str(repo), "show-ref", "--verify", "--quiet", ref], check=False).returncode:
        raise ReleaseError(f"tag does not exist locally: {tag}")
    object_type = git(repo, "cat-file", "-t", ref)
    if object_type != "tag" and not allow_lightweight:
        raise ReleaseError(f"{tag} is a lightweight tag; use an annotated tag for a release")
    commit = resolve_commit(repo, ref)
    head = resolve_commit(repo, "HEAD")
    if commit != head:
        raise ReleaseError(f"HEAD {head} does not match {tag} commit {commit}")
    signature_verified = (
        run(["git", "-C", str(repo), "verify-tag", "--raw", tag], check=False).returncode == 0
    )
    return commit, signature_verified


def create_git_archive(repo: Path, commit: str, destination: Path, prefix: str | None = None) -> None:
    args = ["git", "-C", str(repo), "archive", "--format=tar"]
    if prefix is not None:
        args.extend(["--prefix", prefix])
    args.append(commit)
    with destination.open("wb") as stream:
        run(args, stdout=stream)


def canonical_source_digest(repo: Path, commit: str, scratch: Path) -> str:
    archive = scratch / "canonical-source.tar"
    create_git_archive(repo, commit, archive)
    return sha256_file(archive)


def deterministic_gzip(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output_stream, compresslevel=9, mtime=0) as gz:
            shutil.copyfileobj(input_stream, gz, length=1024 * 1024)


def safe_relative_path(raw: Any, field: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ReleaseError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseError(f"unsafe path in {field}: {raw!r}")
    return path


def evidence_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ReleaseError(f"evidence directory does not exist: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ReleaseError(f"evidence must not contain symbolic links: {path}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ReleaseError(f"evidence must contain only regular files: {path}")
        files.append(path)
    if not files:
        raise ReleaseError("evidence directory is empty")
    return files


def parse_generated_time(raw: Any) -> None:
    if not isinstance(raw, str) or not raw.endswith("Z"):
        raise ReleaseError("generated_at_utc must be an ISO-8601 UTC timestamp ending in Z")
    try:
        dt.datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseError("generated_at_utc is not a valid ISO-8601 timestamp") from exc


def parse_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReleaseError(f"cannot read checksum file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        match = CHECKSUM_LINE_RE.fullmatch(line)
        if not match:
            raise ReleaseError(f"invalid checksum line {path}:{line_number}")
        relative = safe_relative_path(match.group(2), f"{path}:{line_number}").as_posix()
        if relative in checksums:
            raise ReleaseError(f"duplicate checksum entry for {relative} in {path}")
        checksums[relative] = match.group(1).lower()
    if not checksums:
        raise ReleaseError(f"checksum file is empty: {path}")
    return checksums


def scan_sensitive_text(path: Path) -> None:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ReleaseError(f"public evidence file exceeds the 32 MiB scan limit: {path}")
    data = path.read_bytes()
    for label, pattern in SENSITIVE_PATTERNS:
        if pattern.search(data):
            raise ReleaseError(f"possible {label} in public evidence file: {path}")


def run_evidence_verifier(root: Path, repo: Path) -> dict[str, Any]:
    verifier = repo / "tools" / "verify_evidence.py"
    if not verifier.is_file():
        raise ReleaseError(f"evidence verifier is missing: {verifier}")
    result = run(
        [sys.executable, str(verifier), str(root), "--source", str(repo), "--json"]
    )
    try:
        report = json.loads(result.stdout or "")
    except json.JSONDecodeError as exc:
        raise ReleaseError("evidence verifier did not return JSON") from exc
    if not isinstance(report, dict):
        raise ReleaseError("evidence verifier report must be a JSON object")
    return report


def portable_verifier_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove acquisition-machine paths while retaining every verification result."""
    return {key: value for key, value in report.items() if key not in {"run_root", "source"}}


def validate_evidence(
    root: Path,
    *,
    repo: Path,
    commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[tuple[Path, PurePosixPath]]]:
    paths = evidence_files(root)
    manifest_path = root / "manifest.json"
    if manifest_path not in paths:
        raise ReleaseError("evidence directory must contain manifest.json at its root")
    manifest = load_json(manifest_path)
    report = run_evidence_verifier(root, repo)
    if report.get("git_commit") != commit:
        raise ReleaseError(
            f"verified evidence commit {report.get('git_commit')!r} does not match tag commit {commit}"
        )

    evidence_hashes: dict[str, str] = {}
    archive_files: list[tuple[Path, PurePosixPath]] = []
    for path in paths:
        relative = PurePosixPath(path.relative_to(root).as_posix())
        scan_sensitive_text(path)
        evidence_hashes[relative.as_posix()] = sha256_file(path)
        archive_files.append((path, relative))
    return manifest, report, evidence_hashes, archive_files


def create_evidence_tar(
    files: Iterable[tuple[Path, PurePosixPath]],
    destination: Path,
    prefix: str,
) -> None:
    with tarfile.open(destination, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for source, relative in files:
            info = archive.gettarinfo(str(source), arcname=f"{prefix}{relative.as_posix()}")
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
            with source.open("rb") as stream:
                archive.addfile(info, stream)


def safe_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            raise ReleaseError(f"archive contains unsafe path: {member.name}")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise ReleaseError(f"archive contains unsupported member: {member.name}")
    return members


def generate_sbom(source_archive: Path, destination: Path, scratch: Path, required: bool) -> str | None:
    configured_syft = os.environ.get("SGBLAS_SYFT")
    syft = configured_syft or shutil.which("syft")
    if syft is None:
        if required:
            raise ReleaseError("syft is required but is not installed")
        print("warning: syft is not installed; no SBOM was generated", file=sys.stderr)
        return None
    syft_path = Path(syft).expanduser()
    if configured_syft and (not syft_path.is_file() or not os.access(syft_path, os.X_OK)):
        raise ReleaseError(f"SGBLAS_SYFT does not name an executable file: {syft}")
    version_result = run([str(syft_path), "version"])
    syft_version = (version_result.stdout or "").strip()
    if not syft_version:
        raise ReleaseError("syft version output is empty")
    source_dir = scratch / "source-export"
    source_dir.mkdir()
    with tarfile.open(source_archive, mode="r:gz") as archive:
        safe_tar_members(archive)
        archive.extractall(source_dir)
    environment = os.environ.copy()
    environment["SYFT_CHECK_FOR_APP_UPDATE"] = "false"
    with destination.open("wb") as output:
        result = subprocess.run(
            [str(syft_path), f"dir:{source_dir}", "-o", "spdx-json"],
            stdout=output,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseError(f"syft failed to generate an SBOM: {detail}")
    sbom = load_json(destination)
    if not str(sbom.get("spdxVersion", "")).startswith("SPDX-"):
        raise ReleaseError("syft output is not an SPDX JSON document")
    return syft_version


def commit_time(repo: Path, commit: str) -> str:
    raw = git(repo, "show", "-s", "--format=%cI", commit)
    value = dt.datetime.fromisoformat(raw).astimezone(dt.timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def write_checksums(directory: Path, names: Iterable[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in sorted(names)]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package(args: argparse.Namespace) -> None:
    repo = repository_root()
    ensure_clean(repo)
    commit, signature_verified = resolve_release_tag(repo, args.tag, args.allow_lightweight_tag)
    evidence_root = Path(args.evidence_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output == evidence_root or output.is_relative_to(evidence_root) or evidence_root.is_relative_to(output):
        raise ReleaseError("output and evidence directories must be disjoint")
    if output.exists():
        if not output.is_dir():
            raise ReleaseError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise ReleaseError(f"output directory is not empty: {output}")

    with tempfile.TemporaryDirectory(prefix="sgblas-release-") as temporary:
        scratch = Path(temporary)
        source_digest = canonical_source_digest(repo, commit, scratch)
        _, verifier_report, evidence_hashes, files = validate_evidence(
            evidence_root,
            repo=repo,
            commit=commit,
        )

        stem = f"{PROJECT}-{args.tag}"
        source_name = f"{stem}-source.tar.gz"
        evidence_name = f"{stem}-evidence.tar.gz"
        manifest_name = f"{stem}-release-manifest.json"
        verification_name = f"{stem}-evidence-verification.json"
        sbom_name = f"{stem}.spdx.json"

        source_tar = scratch / "source.tar"
        source_archive = scratch / source_name
        create_git_archive(repo, commit, source_tar, prefix=f"{stem}/")
        deterministic_gzip(source_tar, source_archive)

        evidence_tar = scratch / "evidence.tar"
        evidence_archive = scratch / evidence_name
        create_evidence_tar(files, evidence_tar, prefix=f"{stem}-evidence/")
        deterministic_gzip(evidence_tar, evidence_archive)

        sbom_path = scratch / sbom_name
        sbom_generator = generate_sbom(source_archive, sbom_path, scratch, args.require_sbom)
        verification_path = scratch / verification_name
        write_json(
            verification_path,
            {
                "schema_version": 1,
                "project": PROJECT,
                "release_tag": args.tag,
                "git_commit": commit,
                "verifier": "tools/verify_evidence.py",
                "report": portable_verifier_report(verifier_report),
            },
        )
        artifact_paths = [source_archive, evidence_archive, verification_path]
        if sbom_generator is not None:
            artifact_paths.append(sbom_path)
        artifact_entries = {
            path.name: {
                "kind": (
                    "source"
                    if path == source_archive
                    else "evidence"
                    if path == evidence_archive
                    else "verification"
                    if path == verification_path
                    else "sbom"
                ),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        }
        release_manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_tool_version": TOOL_VERSION,
            "project": PROJECT,
            "release_tag": args.tag,
            "git_commit": commit,
            "commit_time_utc": commit_time(repo, commit),
            "annotated_tag_required": not args.allow_lightweight_tag,
            "tag_signature_verified": signature_verified,
            "source_tree_sha256": source_digest,
            "sbom_generator": sbom_generator,
            "evidence_manifest_sha256": evidence_hashes["manifest.json"],
            "evidence_source_sha256": verifier_report.get("source_sha256"),
            "evidence_verification_report": portable_verifier_report(verifier_report),
            "evidence_files": evidence_hashes,
            "artifacts": artifact_entries,
        }
        release_manifest_path = scratch / manifest_name
        write_json(release_manifest_path, release_manifest)
        artifact_paths.append(release_manifest_path)
        write_checksums(scratch, (path.name for path in artifact_paths))

        output.mkdir(parents=True, exist_ok=True)
        for path in artifact_paths:
            shutil.copy2(path, output / path.name)
        shutil.copy2(scratch / "SHA256SUMS", output / "SHA256SUMS")

    verify_directory(output, expected_tag=args.tag, repo=repo)
    print(f"release bundle created at {output}")
    print(f"tag: {args.tag}")
    print(f"commit: {commit}")
    print(f"source_tree_sha256: {source_digest}")


def parse_top_level_checksums(path: Path) -> dict[str, str]:
    checksums = parse_checksum_file(path)
    for name in checksums:
        if PurePosixPath(name).parent != PurePosixPath("."):
            raise ReleaseError(f"release checksum entry must be a top-level file: {name}")
    return checksums


def validate_archive_prefix(path: Path, prefix: str) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        members = safe_tar_members(archive)
        if not members:
            raise ReleaseError(f"archive is empty: {path}")
        root = prefix.rstrip("/")
        expected = root + "/"
        if any(member.name != root and not member.name.startswith(expected) for member in members):
            raise ReleaseError(f"archive member lies outside expected prefix {expected}: {path}")


def files_identical(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(1024 * 1024)
            right_chunk = right_stream.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def verify_evidence_archive(
    path: Path,
    manifest: dict[str, Any],
    prefix: str,
) -> None:
    observed: dict[str, str] = {}
    embedded_manifest: dict[str, Any] | None = None
    with tarfile.open(path, mode="r:gz") as archive:
        for member in safe_tar_members(archive):
            if not member.isfile():
                continue
            if not member.name.startswith(prefix):
                raise ReleaseError(f"evidence archive member lies outside {prefix}: {member.name}")
            relative = member.name[len(prefix) :]
            safe_relative_path(relative, "evidence archive member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ReleaseError(f"cannot read evidence archive member: {member.name}")
            if relative == "manifest.json":
                data = stream.read()
                observed[relative] = hashlib.sha256(data).hexdigest()
                try:
                    value = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ReleaseError("embedded evidence manifest is invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ReleaseError("embedded evidence manifest is not a JSON object")
                embedded_manifest = value
            else:
                observed[relative] = sha256_stream(stream)
    if observed != manifest.get("evidence_files"):
        raise ReleaseError("evidence archive contents do not match the release manifest")
    if embedded_manifest is None:
        raise ReleaseError("evidence archive is missing manifest.json")


def verify_directory(directory: Path, expected_tag: str | None, repo: Path | None) -> None:
    if not directory.is_dir():
        raise ReleaseError(f"release directory does not exist: {directory}")
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ReleaseError("release directory is missing SHA256SUMS")
    checksums = parse_top_level_checksums(checksum_path)
    entries = list(directory.iterdir())
    unsupported = sorted(path.name for path in entries if not path.is_file())
    if unsupported:
        raise ReleaseError("release directory contains non-file entries: " + ", ".join(unsupported))
    actual_files = {path.name for path in entries if path.name != "SHA256SUMS"}
    if actual_files != set(checksums):
        raise ReleaseError("release files do not exactly match SHA256SUMS")
    for name, expected in checksums.items():
        actual = sha256_file(directory / name)
        if actual != expected:
            raise ReleaseError(f"release checksum mismatch: {name}")

    manifests = sorted(directory.glob(f"{PROJECT}-v*-release-manifest.json"))
    if len(manifests) != 1:
        raise ReleaseError("release directory must contain exactly one release manifest")
    manifest = load_json(manifests[0])
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION or manifest.get("project") != PROJECT:
        raise ReleaseError("release manifest schema or project is invalid")
    tag = manifest.get("release_tag")
    commit = manifest.get("git_commit")
    source_digest = manifest.get("source_tree_sha256")
    if not isinstance(tag, str) or not TAG_RE.fullmatch(tag):
        raise ReleaseError("release manifest tag is invalid")
    if expected_tag is not None and tag != expected_tag:
        raise ReleaseError(f"release manifest tag {tag} does not match expected tag {expected_tag}")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseError("release manifest commit is invalid")
    if not isinstance(source_digest, str) or not SHA256_RE.fullmatch(source_digest):
        raise ReleaseError("release manifest source_tree_sha256 is invalid")
    evidence_files_value = manifest.get("evidence_files")
    evidence_report = manifest.get("evidence_verification_report")
    if not isinstance(evidence_files_value, dict) or manifest.get(
        "evidence_manifest_sha256"
    ) != evidence_files_value.get("manifest.json"):
        raise ReleaseError("release manifest does not bind the runner manifest checksum")
    if (
        not isinstance(evidence_report, dict)
        or evidence_report.get("git_commit") != commit
        or manifest.get("evidence_source_sha256") != evidence_report.get("source_sha256")
    ):
        raise ReleaseError("release manifest evidence report is not bound to the tagged commit")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ReleaseError("release manifest artifacts must be an object")
    if set(checksums) != set(artifacts) | {manifests[0].name}:
        raise ReleaseError("SHA256SUMS must contain exactly the declared artifacts and release manifest")
    kinds: dict[str, str] = {}
    for name, metadata in artifacts.items():
        if name not in checksums or not isinstance(metadata, dict):
            raise ReleaseError(f"invalid release artifact entry: {name}")
        if metadata.get("sha256") != checksums[name]:
            raise ReleaseError(f"release manifest checksum mismatch: {name}")
        if metadata.get("size_bytes") != (directory / name).stat().st_size:
            raise ReleaseError(f"release manifest size mismatch: {name}")
        kind = metadata.get("kind")
        if kind not in {"source", "evidence", "verification", "sbom"} or kind in kinds:
            raise ReleaseError(f"invalid or duplicate release artifact kind: {kind}")
        kinds[kind] = name
    if not {"source", "evidence", "verification"}.issubset(kinds):
        raise ReleaseError("release manifest must identify source, evidence, and verification artifacts")

    stem = f"{PROJECT}-{tag}"
    validate_archive_prefix(directory / kinds["source"], f"{stem}/")
    verify_evidence_archive(directory / kinds["evidence"], manifest, f"{stem}-evidence/")
    verification = load_json(directory / kinds["verification"])
    if (
        verification.get("release_tag") != tag
        or verification.get("git_commit") != commit
        or verification.get("report") != manifest.get("evidence_verification_report")
    ):
        raise ReleaseError("evidence verification record does not match the release manifest")
    if "sbom" in kinds:
        if not isinstance(manifest.get("sbom_generator"), str) or not manifest["sbom_generator"]:
            raise ReleaseError("release manifest is missing the SBOM generator version")
        sbom = load_json(directory / kinds["sbom"])
        if not str(sbom.get("spdxVersion", "")).startswith("SPDX-"):
            raise ReleaseError("release SBOM is not SPDX JSON")
    elif manifest.get("sbom_generator") is not None:
        raise ReleaseError("release manifest names an SBOM generator but has no SBOM artifact")

    if repo is not None and expected_tag is not None:
        resolved, _ = resolve_release_tag(repo, expected_tag, manifest.get("annotated_tag_required") is False)
        if resolved != commit:
            raise ReleaseError(f"local tag commit {resolved} does not match release manifest commit {commit}")
        with tempfile.TemporaryDirectory(prefix="sgblas-source-verify-") as temporary:
            scratch = Path(temporary)
            actual_digest = canonical_source_digest(repo, commit, scratch)
            if actual_digest != source_digest:
                raise ReleaseError("local tagged source tree does not match release manifest digest")

            expected_tar = scratch / "expected-source.tar"
            expected_archive = scratch / "expected-source.tar.gz"
            create_git_archive(repo, commit, expected_tar, prefix=f"{stem}/")
            deterministic_gzip(expected_tar, expected_archive)
            supplied_archive = directory / kinds["source"]
            expected_archive_sha256 = sha256_file(expected_archive)
            supplied_archive_sha256 = sha256_file(supplied_archive)
            if (
                expected_archive_sha256 != supplied_archive_sha256
                or not files_identical(expected_archive, supplied_archive)
            ):
                raise ReleaseError(
                    "source archive does not byte-match deterministic git archive "
                    f"for {tag} ({commit}); expected SHA-256 "
                    f"{expected_archive_sha256}, got {supplied_archive_sha256}"
                )
        with tempfile.TemporaryDirectory(prefix="sgblas-evidence-verify-") as temporary:
            extract_root = Path(temporary)
            with tarfile.open(directory / kinds["evidence"], mode="r:gz") as archive:
                safe_tar_members(archive)
                archive.extractall(extract_root)
            report = run_evidence_verifier(extract_root / f"{stem}-evidence", repo)
        if portable_verifier_report(report) != manifest.get("evidence_verification_report"):
            raise ReleaseError("re-extracted evidence verifier report does not match release manifest")
    print(f"verified {tag} release bundle at {directory}")


def verify(args: argparse.Namespace) -> None:
    repo = repository_root()
    ensure_clean(repo)
    verify_directory(Path(args.release_dir).expanduser().resolve(), args.tag, repo)


def source_metadata(args: argparse.Namespace) -> None:
    repo = repository_root()
    ensure_clean(repo)
    commit = resolve_commit(repo, args.commit)
    with tempfile.TemporaryDirectory(prefix="sgblas-source-") as temporary:
        digest = canonical_source_digest(repo, commit, Path(temporary))
    value: dict[str, Any] = {
        "git_commit": commit,
        "source_tree_sha256": digest,
    }
    if args.tag is not None:
        if not TAG_RE.fullmatch(args.tag):
            raise ReleaseError(f"invalid planned release tag: {args.tag}")
        value["release_tag"] = args.tag
    print(json.dumps(value, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser(
        "source-metadata", help="print the clean commit and canonical source-tree SHA-256"
    )
    metadata_parser.add_argument("--commit", default="HEAD", help="commit to describe (default: HEAD)")
    metadata_parser.add_argument("--tag", help="planned SemVer release tag, such as v0.1.0")
    metadata_parser.set_defaults(function=source_metadata)

    package_parser = subparsers.add_parser("package", help="build and verify source/evidence release assets")
    package_parser.add_argument("--tag", required=True, help="existing annotated SemVer release tag")
    package_parser.add_argument("--evidence-dir", required=True, help="complete runner directory rooted at manifest.json")
    package_parser.add_argument("--output-dir", required=True, help="new or empty release output directory")
    package_parser.add_argument(
        "--require-sbom", action="store_true", help="fail instead of omitting the SPDX JSON SBOM when syft is absent"
    )
    package_parser.add_argument(
        "--allow-lightweight-tag",
        action="store_true",
        help="permit a lightweight tag (not recommended; the GitHub workflow does not use this)",
    )
    package_parser.set_defaults(function=package)

    verify_parser = subparsers.add_parser(
        "verify", help="verify checksums, archives, manifests, and the required local tag"
    )
    verify_parser.add_argument("--release-dir", required=True, help="directory containing the release assets")
    verify_parser.add_argument(
        "--tag", required=True, help="bind the bundle to this tag in the current repository"
    )
    verify_parser.set_defaults(function=verify)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except ReleaseError as exc:
        print(f"release error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
