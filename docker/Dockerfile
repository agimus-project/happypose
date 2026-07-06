# syntax=docker/dockerfile:1.7-labs
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system-wide dependencies
RUN apt-get update && \
    apt update && \
    apt install -yq \
        git \
        python3 \
        python3-pip \
        x11-xserver-utils \
        ffmpeg libsm6 libxext6 \
        nvidia-container-runtime \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Change default python version
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv
RUN pip install uv==0.8.24

# Tmp copy of git repo
WORKDIR /happypose
COPY --exclude=docker/ . . 

# Install pypi deps
RUN uv pip install --system --no-build-isolation setuptools==78.1.0 pybind11

RUN uv pip install --system --no-build-isolation -r ./requirements/base.txt
RUN uv pip install --system --no-build-isolation -r ./requirements/pypi.txt
RUN uv pip install --system --no-build-isolation -r ./requirements/cu124.txt

# Install cosypose
RUN pip install -e /happypose/happypose/pose_estimators/cosypose/

# Clean up and envs
WORKDIR ~
RUN rm -r /happypose

CMD ["/bin/bash"]