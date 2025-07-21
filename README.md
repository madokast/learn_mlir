# learn_mlir
学习 MLIR

# 环境准备

```
sudo apt update

sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl \
  libncursesw5-dev xz-utils libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev
```

## python3.11 这个和LLVM最相应

安装 pyenv `curl https://pyenv.run | bash`

按要求添加到 `~/.bashrc`

```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
```

加载 `source ~/.bashrc`

下载 python 3.11.13 `pyenv install 3.11.13`（作废）

注意我们需要编译为动态库 `env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11.13`

查看已安装的版本 `pyenv versions`

全局切换 `pyenv global 3.11.13`

检查 python 版本 `python --version`

使用 `pyenv which python` 查看 python 位置，后续 DPython3_ROOT_DIR 会用到

## onnx-mlir 安装

https://github.com/onnx/onnx-mlir 仓库

https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md 安装手册

安装 ninja `sudo apt install -y ninja-build`

需要 `libprotobuf-dev` 先直接安装试试。失败了，版本过低，可以直接下载预编译版本

```
wget https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protoc-31.1-linux-x86_64.zip
# 解压后添加到 PATH
export PATH=/home/mdk/repo/learn_mlir/protoc-31.1/bin:$PATH

```

还是先要编译 LLVM，具体见手册

```
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DENABLE_LIBOMPTARGET=OFF \
   -DLLVM_ENABLE_LIBEDIT=OFF

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

报错 /usr/bin/ld: /home/codespace/.python/current/lib/libpython3.12.a(import.o): relocation R_X86_64_TPOFF32 against hidden symbol `pkgcontext' can not be used when making a shared object

因为 libpython3 没有动态库，见 pyenv 安装 python

下面安装 onnx-mlir

```
# 指定 MLIR 的位置
git clone --recursive https://github.com/onnx/onnx-mlir.git 
MLIR_DIR=/home/mdk/repo/learn_mlir/llvm-project/build_4_onnx_mlir/lib/cmake/mlir
pythonLocation=/home/mdk/.pyenv/versions/3.11.13
mkdir onnx-mlir/build && cd onnx-mlir/build
cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DPython3_ROOT_DIR=${pythonLocation} \
      -DMLIR_DIR=${MLIR_DIR} \
      ..

cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

编译报错，可能是 GCC 版本太高了（确实啊）
```
sudo apt install gcc-9 g++-9


sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

sudo update-alternatives --config gcc
sudo update-alternatives --config g++

# 恢复

sudo update-alternatives --auto gcc
sudo update-alternatives --auto g++
```

## IREE 安装

进入目录 `iree-install`

在仓库寻找 whl 预编译文件 `https://github.com/iree-org/iree`

下载 `wget https://github.com/iree-org/iree/releases/download/v3.5.0/iree_base_compiler-3.5.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`

直接 `pip install /path/to/the.whl` 安全

# torch 导出 ONNX

进入目录 `cd a01_hello_mlir`

虚拟环境 `python -m venv venv`

激活 `source ./venv/bin/activate`

检测，安装

```
pip install --upgrade pip
pip install torch numpy
pip install onnx onnxscript
pip install onnxruntime
```

运行 `a01_hello_mlir/torch_mod.py` 可以得到 onnx 模型

通过 netron.app 可以查看模型结构

# ONNX 导出 MLIR


