# Third-Party Notices

sgBLAS is licensed under the Apache License, Version 2.0. That license covers
the original material in this repository; it does not relicense external
software, documentation, services, or trademarks.

## NVIDIA CUDA Toolkit and cuBLAS

The CUDA implementation uses interfaces supplied by the NVIDIA CUDA Toolkit,
including the CUDA Runtime and cuBLAS. These are external build and runtime
dependencies. Their source code, libraries, and development tools are not
vendored in this repository and are governed by NVIDIA's applicable license
terms.

The RunPod Dockerfile references an NVIDIA CUDA base image. The image is fetched
separately from its registry and is not distributed as part of this source
repository. Users are responsible for reviewing the terms that apply to the
CUDA Toolkit, cuBLAS, NVIDIA container images, GPU drivers, and any cloud
service they use.

NVIDIA, CUDA, cuBLAS, and CUTLASS are trademarks or product names of NVIDIA
Corporation. Their use here is descriptive and does not imply endorsement.

## NVIDIA CUTLASS

CUTLASS documentation is used as an architectural reference for GEMM
decomposition, pipelining, and scheduling. sgBLAS does not vendor, include,
link against, or otherwise distribute CUTLASS source or binaries. If CUTLASS
code is added in the future, its license and required notices must be preserved
and this file must be updated before distribution.

## Future dependencies and contributions

Contributors must not add third-party code, generated artifacts, datasets, or
model output unless they have confirmed that redistribution is permitted and
have preserved all required copyright, license, patent, attribution, and notice
material. Any new external dependency or incorporated source must be recorded
here or in an accompanying notice file.
