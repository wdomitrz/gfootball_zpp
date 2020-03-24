# Running seed locally (tutorial for ubuntu):
1. This tutorial assumes that you have:
   + Ubuntu 18.04
   + NVIDIA graphic card with compute capability
	 matching  current tensorflow requirements
		- you can check your graphic card compute capability here:
		 https://developer.nvidia.com/cuda-gpus
		- you can check tensorflow requirements here:
		https://www.tensorflow.org/install/gpu
2. Install docker (see Prerequisites -> Docker)
3. Install NVIDIA drivers
+ you can follow tutorial on tensorflow page (recommended):
  https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101
  (please note that this is not the minimal installation)
+ or alternative for minimal installation (not recommended,
  use only if above fails)
```
sudo add-apt-repository ppa:graphics-drivers

sudo apt-get update

sudo apt-get install nvidia-driver-440
```
4. Install nvidia-docker:
   https://github.com/NVIDIA/nvidia-docker#quickstart
   and reboot
5. Now you can continue with: 
   https://github.com/google-research/seed_rl#local-machine-training-on-a-single-level
