from torch.utils.data import DataLoader, Dataset
import torchvision.models.resnet as resnet
from torchvision.datasets import MNIST
import torch.nn as nn
import torch
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

train_dst = MNIST('./', train=True, transform=None, download=True)
train_loader = DataLoader(train_dst, 50, True, num_workers=4, drop_last=True)

parser = argparse.ArgumentParser('debug parser')
parser.add_argument('--local_rank', default=0)

args = parser.parse_args()
local_rank = args.local_rank

model = nn.Sequential(
    resnet.resnet18(),
    nn.Linear(512, 10)
)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(model)

for img, label in train_loader:
    print('len of feature extractor', len(model.module[0]))


