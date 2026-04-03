#!/bin/bash

set -e

OS_TYPE=$(uname)
PYTHON_BIN=${PYTHON_BIN:-python}
PIP_CMD=("$PYTHON_BIN" -m pip)

check_python_compat() {
    "$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info < (3, 11)):
    raise SystemExit(
        "This installer currently targets Python 3.10.x. "
        "Create/activate a Python 3.10 environment first."
    )
PY
}

install_pyopenjtalk() {
    echo "Installing pyopenjtalk from source..."

    VERSION=$(curl -s https://pypi.org/pypi/pyopenjtalk/json | jq -r .info.version)
    TAR_FILE="pyopenjtalk-$VERSION.tar.gz"
    DIR_NAME="pyopenjtalk-$VERSION"

    curl -L "https://files.pythonhosted.org/packages/source/p/pyopenjtalk/$TAR_FILE" -o "$TAR_FILE"
    tar -xzf "$TAR_FILE"
    rm "$TAR_FILE"

    CMAKE_FILE="$DIR_NAME/lib/open_jtalk/src/CMakeLists.txt"
    if [[ "$OS_TYPE" == "Darwin"* ]]; then
        sed -i '' -E 's/cmake_minimum_required\(VERSION[^\)]*\)/cmake_minimum_required(VERSION 3.5...3.31)/' "$CMAKE_FILE"
    else
        sed -i -E 's/cmake_minimum_required\(VERSION[^\)]*\)/cmake_minimum_required(VERSION 3.5...3.31)/' "$CMAKE_FILE"
    fi

    # pyopenjtalk 的 sdist 里只有 .pyx，没有预生成的 .cpp。
    # 这里提前把构建依赖装进当前环境，确保 setuptools 会走 cythonize，而不是错误地去找缺失的 .cpp。
    "${PIP_CMD[@]}" install "cython>=0.29.16" "setuptools_scm>=8"

    # 直接从补丁后的源码目录安装，避免在 macOS 上重新打包后被 pip 误判成无效源码包。
    "${PIP_CMD[@]}" install --no-build-isolation "./$DIR_NAME"
    rm -rf "$DIR_NAME"
}

check_python_compat

# conda login shell 有时会把 PATH 重置回 base，显式把当前环境的 bin 提前，确保 cmake/pip 指向激活环境。
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

# 兼容性改造后的 macOS 路径单独处理：
# 1. Apple Silicon / Intel Mac 不走 CUDA / ROCm 检测。
# 2. 先装 CPU/MPS 可用的 PyTorch，再装其余依赖。
# 3. portaudio 放在这里安装，避免后面 pyaudio 编译时直接失败。
if [[ "$OS_TYPE" == "Darwin"* ]]; then
    echo "macOS detected. Installing macOS-compatible dependencies..."

    conda install ffmpeg cmake jq pkg-config portaudio -y
    hash -r

    echo "Installing PyTorch for macOS (CPU/MPS)..."
    "${PIP_CMD[@]}" install torch torchvision torchaudio

    install_pyopenjtalk
    "${PIP_CMD[@]}" install -r extra-req.txt --no-deps
    "${PIP_CMD[@]}" install -r requirements.txt

    echo "macOS installation completed successfully!"
    exit 0
fi

# 安装构建工具
# Install build tools
echo "Installing GCC..."
conda install -c conda-forge gcc=14 -y

echo "Installing G++..."
conda install -c conda-forge gxx -y

echo "Installing ffmpeg and cmake..."
conda install ffmpeg cmake -y

# 设置编译环境
# Set up build environment
export CMAKE_MAKE_PROGRAM="$CONDA_PREFIX/bin/cmake"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"

echo "Checking for CUDA installation..."
if command -v nvidia-smi &>/dev/null; then
    USE_CUDA=true
    echo "CUDA found."
else
    echo "CUDA not found."
    USE_CUDA=false
fi

if [ "$USE_CUDA" = false ]; then
    echo "Checking for ROCm installation..."
    if [ -d "/opt/rocm" ]; then
        USE_ROCM=true
        echo "ROCm found."
        if grep -qi "microsoft" /proc/version; then
            echo "You are running WSL."
            IS_WSL=true
        else
            echo "You are NOT running WSL."
            IS_WSL=false
        fi
    else
        echo "ROCm not found."
        USE_ROCM=false
    fi
fi

if [ "$USE_CUDA" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
elif [ "$USE_ROCM" = true ]; then
    echo "Installing PyTorch with ROCm support..."
    "${PIP_CMD[@]}" install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
else
    echo "Installing PyTorch for CPU..."
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 cpuonly -c pytorch
fi

echo "Installing Python dependencies from requirements.txt..."

# 刷新环境
# Refresh environment
hash -r

# pyopenjtalk Installation
conda install jq -y
install_pyopenjtalk

"${PIP_CMD[@]}" install -r extra-req.txt --no-deps

"${PIP_CMD[@]}" install -r requirements.txt

if [ "$USE_ROCM" = true ] && [ "$IS_WSL" = true ]; then
    echo "Update to WSL compatible runtime lib..."
    location=$("${PIP_CMD[@]}" show torch | grep Location | awk -F ": " '{print $2}')
    cd "${location}"/torch/lib/ || exit
    rm libhsa-runtime64.so*
    cp /opt/rocm/lib/libhsa-runtime64.so.1.2 libhsa-runtime64.so
fi

echo "Installation completed successfully!"
