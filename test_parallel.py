from torch.utils.data import DataLoader, Dataset
import torchvision.models.resnet as resnet
from torchvision.datasets import MNIST
import torch.nn as nn
import torch
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import torch.distributed as dist
#train_dst = MNIST('./', train=True, transform=None, download=True)
#train_loader = DataLoader(train_dst, 50, True, num_workers=4, drop_last=True)

parser = argparse.ArgumentParser('debug parser')
parser.add_argument('--local_rank', default=0, type=int)

args = parser.parse_args()
local_rank = args.local_rank

torch.cuda.set_device(int(local_rank))
dist.init_process_group(backend='nccl', init_method='env://')

model = nn.Sequential(
    resnet.resnet18(),
    nn.Linear(512, 10)
).cuda()
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(model)

data = torch.randn((3000, 3, 32, 32)).cuda()
class SDST(Dataset):
    def __init__(self, dat):
        super(SDST, self).__init__()
        self.x =dat

    def __getitem__(self, item):
        return self.x[item]
    
    def __len__(self):
        return len(self.x)
train_dst = SDST()
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(train_dst)
train_loader = DataLoader(train_dst, 50, True, num_workers=4, sampler=sampler,pin_memory=True, drop_last=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for img in data:
    prd = model(img)
    print('len of feature extractor', len(model.module[0]))


