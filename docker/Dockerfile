# syntax=docker/dockerfile:1.7-labs
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system-wide dependencies
RUN apt-get update && \
    apt install -yq software-properties-common && \
    apt update && \
    apt install -yq \
        git \
        python3 \
        python3-pip \
        ffmpeg libsm6 libxext6 \
        nvidia-container-runtime \
        libgl1 libegl1-mesa libgles2-mesa \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Change default python version
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv
RUN pip install uv==0.8.24

# Tmp copy of repo's root
WORKDIR /happypose
COPY --exclude=docker/ . . 

# Install pypi deps
RUN uv sync --extra pypi --extra cu124

CMD ["/bin/bash"]