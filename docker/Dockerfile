# syntax=docker/dockerfile:1.7-labs
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system-wide dependencies
RUN apt-get update && \
    apt install -yq software-properties-common && \
    add-apt-repository -y ppa:mozillateam/ppa && \
    apt update && \
    apt install -yq \
        curl \
        nano \
        git \
        python3 \
        python3-pip \
        p7zip-full \
        ffmpeg libsm6 libxext6 \
        nvidia-container-runtime \
        libgl1 libegl1-mesa libgles2-mesa \
        firefox=153.0.1+build2-0ubuntu0.22.04.1~mt1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Change default python version
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv
RUN pip install uv==0.8.24

# Tmp copy of repo's root
WORKDIR /happypose
COPY . .

# Install pypi deps
RUN uv sync --extra pypi --extra cu124

# Install geckodriver
RUN curl -LO "https://github.com/mozilla/geckodriver/releases/download/v0.37.1/geckodriver-v0.37.1-linux64.tar.gz"
RUN mkdir -p /opt/geckodriver && tar -xvf geckodriver-v0.37.1-linux64.tar.gz -C /opt/geckodriver
ENV PATH="$PATH:/opt/geckodriver"

# Create unpriviledged user
RUN useradd -ms /bin/bash happypose
USER happypose

CMD ["/bin/bash"]
