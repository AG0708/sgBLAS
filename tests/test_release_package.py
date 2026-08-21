#!/usr/bin/env python3
"""Regression tests for fail-closed sgBLAS release packaging."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from unittest import mock, TestCase, main


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))
import release_package  # noqa: E402


class SensitiveEvidenceScanTests(TestCase):
    def test_nul_prefixed_token_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "binary.bin"
            artifact.write_bytes(b"\x00ELF\x00ghp_" + b"A" * 40)
            with self.assertRaisesRegex(release_package.ReleaseError, "GitHub token"):
                release_package.scan_sensitive_text(artifact)

    def test_file_over_scan_limit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "oversized.bin"
            with artifact.open("wb") as stream:
                stream.truncate(32 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(release_package.ReleaseError, "exceeds the 32 MiB"):
                release_package.scan_sensitive_text(artifact)


class SourceArchiveBindingTests(TestCase):
    def git(self, repo: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    def test_self_consistent_substitute_source_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            evidence = root / "evidence"
            output = root / "release"
            (repo / "tools").mkdir(parents=True)
            evidence.mkdir()
            (repo / "README.md").write_text("trusted source\n", encoding="utf-8")
            (repo / "tools" / "verify_evidence.py").write_text(
                """#!/usr/bin/env python3
import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("run_root")
parser.add_argument("--source")
parser.add_argument("--json", action="store_true")
args = parser.parse_args()
manifest = json.load(open(args.run_root + "/manifest.json", encoding="utf-8"))
print(json.dumps({
    "run_root": args.run_root,
    "source": args.source,
    "source_sha256": manifest["source_sha256"],
    "git_commit": manifest["git_commit"],
    "variants": ["wide", "hybrid"],
    "shapes": [[256, 256, 256]],
    "independent_logs_per_variant": 6,
    "sanitizers": ["memcheck", "racecheck", "initcheck", "synccheck"],
    "artifact_count": 1,
}))
""",
                encoding="utf-8",
            )
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "Release Test")
            self.git(repo, "config", "user.email", "release@example.invalid")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-qm", "trusted release")
            commit = self.git(repo, "rev-parse", "HEAD")
            self.git(repo, "tag", "-a", "v0.1.0", "-m", "v0.1.0")

            (evidence / "manifest.json").write_text(
                json.dumps({"git_commit": commit, "source_sha256": "a" * 64}) + "\n",
                encoding="utf-8",
            )
            (evidence / "summary.json").write_text('{"ok": true}\n', encoding="utf-8")
            arguments = argparse.Namespace(
                tag="v0.1.0",
                evidence_dir=str(evidence),
                output_dir=str(output),
                require_sbom=False,
                allow_lightweight_tag=False,
            )
            with (
                mock.patch.object(release_package, "repository_root", return_value=repo),
                mock.patch.object(release_package.shutil, "which", return_value=None),
                mock.patch.dict(os.environ, {"SGBLAS_SYFT": ""}, clear=False),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                release_package.package(arguments)

            release_manifest_path = next(output.glob("*-release-manifest.json"))
            release_manifest = release_package.load_json(release_manifest_path)
            source_name = next(
                name
                for name, record in release_manifest["artifacts"].items()
                if record["kind"] == "source"
            )
            source_archive = output / source_name
            raw_tar = root / "substitute.tar"
            payload = b"attacker-controlled source\n"
            with tarfile.open(raw_tar, "w", format=tarfile.PAX_FORMAT) as archive:
                member = tarfile.TarInfo("sgblas-v0.1.0/README.md")
                member.size = len(payload)
                member.mode = 0o644
                member.mtime = 0
                archive.addfile(member, io.BytesIO(payload))
            release_package.deterministic_gzip(raw_tar, source_archive)

            source_record = release_manifest["artifacts"][source_name]
            source_record["sha256"] = release_package.sha256_file(source_archive)
            source_record["size_bytes"] = source_archive.stat().st_size
            release_package.write_json(release_manifest_path, release_manifest)
            release_package.write_checksums(
                output,
                (path.name for path in output.iterdir() if path.name != "SHA256SUMS"),
            )

            with self.assertRaisesRegex(
                release_package.ReleaseError,
                "source archive does not byte-match deterministic git archive",
            ):
                release_package.verify_directory(output, "v0.1.0", repo)


if __name__ == "__main__":
    main()
