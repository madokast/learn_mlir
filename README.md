# learn_mlir
学习 MLIR

# 环境准备

sudo apt update

## python3.11

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

## torch 导出 ONNX

虚拟环境 `python -m venv venv`

激活 `source ./venv/bin/activate`

检测，安装

```
pip3 install torch torchvision torchaudio
pip install --upgrade onnx onnxscript
```

运行 `a01_hello_mlir/torch_mod.py` 可以得到 onnx 模型

通过 netron.app 可以查看模型结构

## ONNX 导出 MLIR


