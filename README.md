# learn_mlir
学习 MLIR

# 外部仓库

## torch-mlir

git clone https://github.com/llvm/torch-mlir.git

## llvm-project

git clone --recursive https://github.com/llvm/llvm-project.git

cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;mlir;polly" -G "Unix Makefiles" ../llvm


