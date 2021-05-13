#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=bp1119 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/bohua/miniconda3/bin/:$PATH
source activate
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train_v4.py
/usr/bin/nvidia-smi
uptime