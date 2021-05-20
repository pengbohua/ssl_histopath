#!/bin/bash
export PATH=miniconda3/bin/:$PATH
export PATH=/vol/bitbucket/bohua/miniconda3/bin/:$PATH
source activate
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 test_parallel.py
/usr/bin/nvidia-smi
