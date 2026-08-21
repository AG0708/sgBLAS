ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/sgblas

CMD ["/bin/bash"]
