# sgBLAS v0.1.0 release checklist

This release is evidence-gated. Existing pre-publication A100 logs are not
release evidence: they record `git_commit=uncommitted`, and source changed after
measurement. Generate a fresh self-contained runner directory from the exact
release commit.

## 1. Freeze the source

- [ ] Land source, CI, licensing, provenance, security, and documentation in a
  reviewed commit; require an empty `git status --short` and green CI.
- [ ] Review tracked and ignored files for credentials, copied code, generated
  output, personal paths, hostnames, GPU UUIDs, and other stable identifiers.
- [ ] Record the commit and canonical archive digest:

  ```bash
  python3 tools/release_package.py source-metadata --commit HEAD --tag v0.1.0
  ```

  The digest is SHA-256 over uncompressed `git archive --format=tar <commit>`.

## 2. Acquire and verify A100 evidence

- [ ] Check out the recorded commit on the A100 host with a clean worktree.
- [ ] Run `tools/run_a100_tuning.py` to completion. Keep its whole output
  directory, rooted at `manifest.json`; do not select or rewrite individual
  files. It includes both variants' raw logs, summaries, build logs, CMake
  cache, and exact tested binaries under each `tested-binaries/` folder.
- [ ] Copy the directory away from the build host and prove it is portable and
  still bound to this checkout:

  ```bash
  python3 tools/verify_evidence.py /absolute/path/to/run-root --source "$PWD"
  ```

- [ ] Review the complete directory before publication. Redact sensitive data
  at acquisition time and rerun; never mutate a completed manifest or log. The
  packager also fails on common secret formats, private keys, and NVIDIA GPU
  UUIDs.

## 3. Tag and package

- [ ] Only after evidence verification succeeds, create an annotated tag on
  that exact tested commit (prefer a signed tag):

  ```bash
  git tag -s v0.1.0 <tested-commit> -m "sgBLAS v0.1.0"
  ```

  Use `git tag -a` only when signing is unavailable. Never move a published
  tag; issue v0.1.1 for corrections.
- [ ] Package from a clean checkout of that tag:

  ```bash
  python3 tools/release_package.py package \
    --tag v0.1.0 \
    --evidence-dir /absolute/path/to/run-root \
    --output-dir out/release-v0.1.0

  python3 tools/release_package.py verify \
    --release-dir out/release-v0.1.0 \
    --tag v0.1.0
  ```

  `syft` is optional locally. Add `--require-sbom` to make SPDX generation a
  hard gate. Expected assets are deterministic source and evidence archives,
  an evidence-verification report, release manifest, optional SPDX JSON, and
  `SHA256SUMS`.

## 4. Review, then publish

The `Release bundle` workflow accepts a same-repository workflow artifact whose
root is the complete runner directory and contains `manifest.json`. Run it with
`publish=false`, inspect the resulting assets, and only then rerun with
`publish=true`. Actions and Syft setup are pinned to immutable revisions; the
publish job reverifies everything and refuses to replace an existing release.

For evidence acquired outside GitHub Actions, use the local package flow above.
After pushing the exact commit and annotated tag, publish only the verified
directory:

```bash
gh release create v0.1.0 out/release-v0.1.0/* \
  --generate-notes --title "sgBLAS v0.1.0" --verify-tag
```

## 5. Verify the public release

- [ ] Download assets into an empty directory and run
  `sha256sum --check SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS` on macOS).
- [ ] Run `release_package.py verify --release-dir ... --tag v0.1.0` from a
  clean checkout; this extracts the evidence and reruns the strict verifier.
- [ ] Confirm the public tag resolves to the manifest commit and that headline
  performance claims exactly match the tagged A100 evidence and FP32 corpus.
