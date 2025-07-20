# learn_mlir
学习 MLIR

# 环境准备

sudo apt update

## python3.11 可跳过直接用3.12

安装 pyenv `curl https://pyenv.run | bash`

按要求添加到 `~/.bashrc`

```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
```

加载 `source ~/.bashrc`

下载 python 3.11.13 `pyenv install 3.11.13`

查看已安装的版本 `pyenv versions`

全局切换 `pyenv global 3.11.13`

检查 python 版本 `python --version`

## onnx-mlir 安装

https://github.com/onnx/onnx-mlir 仓库

https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md 安装手册

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

运行 `a01_hello_mlir/onnx_infer.py` 用 onnx 进行推理

# ONNX 导出 MLIR


