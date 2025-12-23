#!/bin/bash
set -euxo pipefail

cd csrc
rm -rf build
mkdir -p build
cd build

export TORCH_CUDA_ARCH_LIST="8.0+PTX"
cmake -DCMAKE_PREFIX_PATH=/opt/conda/envs/Quant/lib/python3.9/site-packages/torch \
   -DQUANT_TORCH_HOME=/opt/conda/envs/Quant/lib/python3.9/site-packages/torch \
   -DCMAKE_BUILD_TYPE=Release \
   -DQUANT_CUDA_HOME=/usr/local/cuda  \
   -DQUANT_CUDNN_HOME=/opt/conda/envs/Quant/lib/python3.9/site-packages/nvidia/cudnn/lib ..
make -j
cp libQuant_kernels.so ../../Quantization/Quant_kernels.so

cd ../../
