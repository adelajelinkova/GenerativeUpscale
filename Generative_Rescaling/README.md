# Generative Upscaling
The Generative Upscaling method trains a downscaler and upscaler and can be used to create high resolution images specially trained for your dataset. It is based on the original Invertible Rescaling Network by splitting the downscaler and upscaler.

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [PyTorch >= 1.0](https://pytorch.org/) with CUDA enabled
- windows paging set to 12000-24000
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`
  
## Dataset Preparation
Commonly used training and testing datasets can be downloaded [here](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md).


# Acknowledgement
This code is based on the paper: "Invertible Rescaling Network and Its Extensions". \[[link](https://link.springer.com/article/10.1007/s11263-022-01688-4)\]\[[arxiv](https://arxiv.org/abs/2210.04188)\]. 