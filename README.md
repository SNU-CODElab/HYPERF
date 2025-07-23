# HYPERF: Environment Setup and Build Guide
This document provides a step-by-step guide to setting up a reproducible HYPERF development environment using Docker.
You’ll find instructions on building the Docker image, installing dependencies, compiling required components from source, and running experiments.
The latest Dockerfile is also available at docker/Dockerfile in this repository.

## 1. Setting Up the Docker Environment
We recommend using Docker to ensure a consistent build and runtime environment for HYPERF.
The provided Dockerfile installs all essential system dependencies on top of Ubuntu 20.04.

**Example** `Dockerfile`:
```dockerfile
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
```
> You can also find this file at `docker/Dockerfile` in the repository.

### Building and Running the Docker Container
Use the following commands to build the Docker image and launch the container:
```bash
docker build -t hpdc-artifact -f Dockerfile .
docker run -dit --name hpdc-artifact --restart=always --privileged hpdc-artifact /bin/bash
docker exec -it hpdc-artifact bash
```

## 2. Installing Dependencies
Once inside the container, install the required Python packages:
```bash
pip3 install tornado psutil 'xgboost==1.5.0' cloudpickle attrs decorator numpy typing_extensions pytest pygments
```

Some components require a higher version of CMake (3.31.8). Install it as follows:
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.tar.gz
tar -xzvf cmake-3.31.8-linux-x86_64.tar.gz
cp -a cmake-3.31.8-linux-x86_64/* /usr/local
```
> Ensure that your system CMake version does not conflict with this installation.

## 3. Cloning the HYPERF Source Code and Setting Up the Environment
Clone the main repository, set the environment variable, and initialize submodules:
```bash
git clone https://github.com/SNU-CODElab/HYPERF.git
export HYPERF_HOME=/root/HYPERF
cd $HYPERF_HOME
git submodule update --init --recursive
```
> The `HYPERF_HOME` environment variable is used throughout the build process.

## 4. Building TVM
TVM is a core dependency for HYPERF. Build it as follows:

```bash
cd $HYPERF_HOME/tvm
mkdir build
cp cmake/config_llvm.cmake build/config.cmake
cd build
cmake -G Ninja ../
ninja
```

After building, configure the TVM-related environment variables:
```bash
export TVM_HOME=$HYPERF_HOME/tvm
export PYTHONPATH=$HYPERF_HOME/tvm/python
export LD_LIBRARY_PATH=$TVM_HOME/build:$LD_LIBRARY_PATH
```

Create a symbolic link to the TVM directory:
```bash
cd $HOME
ln -s $HYPERF_HOME/tvm tvm
```

## 5. Building and Installing LLVM
LLVM (with Clang and OpenMP support) must be built from source:
```bash
cd $HYPERF_HOME/llvm-project
mkdir build
cd build
cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX=/usr/local/ \
  -DLLVM_ENABLE_PROJECTS="clang;openmp" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  ../llvm
ninja
ninja install
```

After installation, update the `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/usr/local/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH
```

## 6. Rebuilding TVM (with LLVM and OpenMP)
Now, rebuild TVM so that it links with the newly built OpenMP library:

```bash
cd $HYPERF_HOME/tvm
mv build build-llvm
mkdir build
cp cmake/config.cmake build/config.cmake
cd build
cmake -G Ninja \
  -DOMP_LIBRARY=/usr/local/lib/x86_64-unknown-linux-gnu/libomp.so \
  ../
ninja
```

## 7. Building TVM-HPC
Build the TVM-HPC:
```bash
cd $HYPERF_HOME/TVM_HPC
mkdir build
cd build
cmake -G Ninja ../
ninja
```

Create a symbolic link to the TVM directory:
```
cd $HOME
ln -s $HYPERF_HOME/TVM_HPC TVM_HPC
```

## 8. Running Experiments
To run the provided experiment suite, use:
```bash
cd $HYPERF_HOME
./run_all.sh
# MARCH=sapphirerapids ./run_all.sh
```
> Modify `run_all.sh` as needed.

### Running Custom Experiments
You can run custom experiments with your programs:
```bash
cd $HYPERF_HOME
./build_script.sh <args>
```
> Replace `<args>`

## Tip
To avoid having to set environment variables every time, you can add them to your `.bashrc` or `.profile` file.
```bash
export HYPERF_HOME=/root/HYPERF
export TVM_HOME=$HYPERF_HOME/tvm
export PYTHONPATH=$HYPERF_HOME/tvm/python
export LD_LIBRARY_PATH=/usr/local/lib/x86_64-unknown-linux-gnu:$TVM_HOME/build:$LD_LIBRARY_PATH
```
