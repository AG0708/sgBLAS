# Provenance

## Authorship and development process

sgBLAS is a project of Abhinav Gorrepati. Its pre-publication development used
AI coding assistants, including OpenAI Codex, for activities that may have
included planning, implementation drafts, tests, documentation, review, and
debugging. Abhinav selected, reviewed, integrated, and takes responsibility for
the material released in this repository.

This disclosure does not claim that every line was independently handwritten.
AI-generated suggestions are treated as untrusted drafts: they do not establish
originality, correctness, or license compatibility, and they must receive the
same human review as any other contribution.

The public Git history begins with the first reviewed release snapshot. Earlier
intermediate commits are not available as an authorship record. The phrase
"from scratch" means that no third-party SGEMM implementation is vendored or
linked into sgBLAS itself; it does not mean that the project was developed
without documentation, architectural references, standard APIs, or AI
assistance.

## External-source policy

The project uses NVIDIA's public CUDA, cuBLAS, PTX, and CUTLASS documentation as
technical references. CUTLASS is reference-only and no CUTLASS code is vendored.
External code may be incorporated only when its provenance and license are
known, redistribution is compatible with this project, and all required notices
are preserved in `THIRD_PARTY_NOTICES.md` or another appropriate notice file.

Contributors are responsible for identifying any copied, adapted, translated,
or generated material in their changes. When provenance is uncertain, the
material must not be merged until ownership and licensing are resolved.

## Source and evidence binding policy

A correctness, sanitizer, or performance claim is considered reproducible
release evidence only when its artifact bundle records all of the following:

- the exact Git commit SHA from a clean working tree;
- a deterministic source-tree digest and hashes of the tested binaries;
- the container image digest or complete native toolchain and operating-system
  versions;
- GPU model, compute capability, driver, CUDA runtime, CUDA compiler, and cuBLAS
  versions, with sensitive stable identifiers removed from public artifacts;
- the exact commands, configuration flags, seed, warmup count, repeat count,
  cache policy, and relevant environment variables;
- raw correctness, sanitizer, and benchmark logs plus a checksum manifest; and
- a generated summary whose values can be traced back to those raw logs.

Any source, build configuration, toolchain, or measurement-protocol change
breaks that binding. Affected claims must be rerun or explicitly labeled as
historical evidence for a different snapshot.

Benchmark and sanitizer results created before the first public commit were
measured from an uncommitted development snapshot. They may be useful as
historical development evidence, but they are not release-reproducible evidence
for a later commit until the full validation campaign is rerun and bound as
described above.
