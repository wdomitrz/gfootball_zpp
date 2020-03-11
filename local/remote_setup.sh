#!/bin/bash
sudo apt-get update;
sudo apt-get install -y linux-headers-$(uname -r);
sudo apt-get install -y gcc;

sudo add-apt-repository ppa:graphics-drivers;

sudo apt-get install -y nvidia-driver-440;

sudo apt-get install -y nvidia-cuda-dev;

sudo apt-get install -y nvidia-utils-440;

sudo dpkg -i nccl-repo-ubuntu1804-2.5.6-ga-cuda10.2_1-1_amd64.deb;

sudo apt-get update && sudo apt-get install -y git apt-transport-https ca-certificates curl gnupg2 software-properties-common;

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -;

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable";

sudo apt-get update;
sudo apt-get install -y docker-ce docker-ce-cli containerd.io;

sudo usermod -aG docker $USER;

# taken from nvidia repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
# in case of errors see http://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/


sudo systemctl restart docker

sudo systemctl enable docker;
