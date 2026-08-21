# Security Policy

## Supported versions

sgBLAS is currently pre-1.0. Security fixes are made only on the latest public
release and the current `main` branch. Older releases, development snapshots,
and locally modified builds are not supported unless a maintainer states
otherwise.

## Reporting a vulnerability

Please use GitHub Private Vulnerability Reporting for this repository when it is
available. Do not open a public issue for an unpatched vulnerability, a real
credential, or sensitive machine information. If private reporting is not
enabled, contact the repository owner through their GitHub profile and request
a private reporting channel before sending details.

Include, where possible:

- the affected commit, release, platform, GPU, driver, and CUDA versions;
- a minimal reproduction and the security impact;
- relevant sanitizer output or logs, with credentials, tokens, hostnames,
  hardware UUIDs, and other stable identifiers removed; and
- whether the issue is already public or has been reported elsewhere.

Do not use production secrets or third-party systems when demonstrating an
issue. Reports are handled on a best-effort basis; there is currently no bug
bounty or guaranteed response-time service level.

## Coordinated disclosure

The maintainer will attempt to reproduce the report, assess affected versions,
prepare a fix, and agree on a disclosure timeline with the reporter. Please
allow a reasonable remediation period before public disclosure. Credit will be
given when requested and legally permitted.

## Operational safety

- Install CUDA, GPU drivers, container images, and cloud tooling only from
  trusted sources, and prefer immutable image digests for reproducible runs.
- Run repository scripts only from a reviewed checkout and a trusted
  environment. Do not pass untrusted environment-variable values to build or
  benchmarking commands.
- Review generated build, benchmark, profiler, and machine-probe artifacts
  before sharing them. They can contain absolute paths, host metadata, stable
  device identifiers, or other information that does not belong in a public
  report.
- Never commit credentials. Treat ignore rules as defense in depth rather than
  a substitute for secret scanning and staged-diff review.
