#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=bp1119 # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH -w cloud-vm-
export PATH=/vol/bitbucket/bohua/miniconda3/bin/:$PATH
source activate
CUDA_VISIBLE_DEVICES=0,1 python -m  torch.distributed.launch -nproc_per_node=2 train_easy.py --batch_size=100 --results-dir='./result_easy'
/usr/bin/nvidia-smi
uptime