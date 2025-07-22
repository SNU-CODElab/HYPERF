FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    python3-dev \
                    python3-pip \
                    ca-certificates \
                    g++ \
                    gcc \
                    make \
                    git \
                    aria2 \
                    wget \
                    build-essential \
                    xutils-dev \
                    bison \
                    zlib1g-dev \
                    flex \
                    libglu1-mesa-dev \
                    git \
                    libssl-dev \
                    libxml2-dev \
                    libboost-all-dev \
                    vim \
                    ninja-build \
                    bc \
                    git-lfs \
                    libtinfo-dev \
                    htop \
                    libedit-dev
ENV HOME /root
WORKDIR /root
CMD ["/bin/bash"]
