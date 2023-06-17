import torch
import torchvision
import sys
import tensorflow as tf
import os

#检查 PyTorch 和 Torchvision 等版本
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA version:", torch.version.cuda)
print("Python version:", sys.version)
print("cuDNN version:",torch.backends.cudnn.version()) # type: ignore


#检查 GPU 是否可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
version = tf.__version__  # 输出 tensorflow 版本
gpu_ok = len(tf.config.list_physical_devices('GPU')) > 0  # 输出 gpu 可否使用（True/False）
print("tf version:", version, "\nuse GPU:", gpu_ok)
tf.test.is_built_with_cuda()  # 判断 CUDA 是否可用（True/False）
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#version reference:
# https://tensorflow.google.cn/install/source?hl=en  #tf version
# https://pytorch.org/get-started/previous-versions/ #torch,torchvision,cuda version

# update cuda version to CUDA 11.1:
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
