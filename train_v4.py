from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import numpy as np
import os
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BasicDataset(Dataset):
    def __init__(self, img_file, labels, transform=None, train=True):
        super(BasicDataset, self).__init__()
        self.x = img_file['x'][:]
        self.y = torch.from_numpy(labels).long()
        self.length = self.y.shape[0]
        self.transform = transform
        self.targets = self.y
        self.num_classes = 2
        self.train = train

    def __getitem__(self, item):

        x = torch.from_numpy(self.x[item]).float()
        y = self.y[item]
        x = x.permute(2, 0, 1)  # C H W

        if self.train:
            assert isinstance(self.transform, list), 'transforms must include positive and negative transforms'
            transform_p = self.transform[0]
            transform_n = self.transform[1]
            img_q = transform_p(x)
            img_p = transform_p(x)
            img_n = transform_n(x)
            return img_q, img_p, img_n
        else:
            img1 = self.transform(x)
            return img1, y

    def __len__(self):
        return self.length


class Subset(Dataset):
    """
    Subset of the Pathology dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]][0], self.dataset.targets[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            # data is shuffled along batch dimension
            # batch normalize separately on each GPU
            # batch_norm changes running_mean 0, 1, 2 for gpu0 3, 4, 5 for gpu1. note that 0 and 3 are means for ch1
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, feature_dim=128, arch=None, bn_splits=16, mlp=True):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
                if mlp:
                    dim_mlp = module.weight.shape[-1]
                    self.net.append(nn.Sequential(
                        nn.Linear(dim_mlp, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True))
                    )

            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        return x


class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True, mlp=False):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits, mlp=mlp)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits, mlp=mlp)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, img_k):
        if isinstance(img_k, list):
            img_p, img_n = img_k
        else:
            raise ValueError('key images must include both positive and negative pair')
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        positive_idx = img_p.shape[0]
        _img_k = torch.cat([img_p, img_n], dim=0)
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            _img_k, idx_unshuffle = self._batch_shuffle_single_gpu(_img_k)

            _img_k = self.encoder_k(_img_k)  # keys: 2NxC
            _img_k = F.normalize(_img_k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(_img_k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        k_p = k[:positive_idx]  # NxC
        k_n = k[positive_idx:]  # NxC
        l_pos = torch.einsum('nc,nc->n', [q, k_p]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_1 = torch.einsum('nc, nc->n', [q, k_n]).unsqueeze(-1)
        l_neg_2 = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+1+K)
        logits = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = LabelSmoothingCrossEntropy().cuda()(logits, labels)

        return loss, q, k_p, k_n

    def forward(self, img_q, img_pos, img_neg):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k_p_2, k_n_2 = self.contrastive_loss(img_q, [img_pos, img_neg])
            loss_21, q2, k_p_1, k_n_1 = self.contrastive_loss(img_pos, [img_q, img_neg])
            loss = loss_12 + loss_21
            k_n = torch.cat([k_n_1, k_n_2], dim=0)
        else:  # asymmetric loss
            loss, q, k_p, k_n = self.contrastive_loss(img_q, [img_pos, img_neg])

        self._dequeue_and_enqueue(k_n)

        return loss


# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)
    data_loader.sampler.set_epoch(epoch)

    total_loss, total_num = 0.0, 0
    for im_1, im_2, img_n in data_loader:
        im_1, im_2, img_n = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True), img_n.cuda(non_blocking=True)
        loss = net(im_1, im_2, img_n)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size

    return total_loss / total_num


# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        if args.warmup_epochs:
            lr *= epoch / args.warmup_epochs if epoch < args.warmup_epochs else 0.5 * (
                        1. + math.cos(math.pi * epoch / args.epochs))
        else:
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = 2
    # classes = args.num_classes
    # initialize
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(
            memory_data_loader.dataset.dataset.targets[memory_data_loader.dataset.indices].squeeze(),
            device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)
            total_num += data.size(0)
            total_top1 += (pred_labels.argmax(dim=1, keepdim=True) == target).float().sum().item()

    return total_top1 / len(test_data_loader.dataset) * 100


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def linear_finetune(num_epo, net, data_loader, train_optimizer, train=True, linear_temperature=0.1):
    is_train = train
    net.train() if is_train else net.eval()
    net = net.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    for i in range(1, num_epo + 1):
        with (torch.enable_grad() if is_train else torch.no_grad()):
            for sample in data_loader:
                img, target = sample
                img = img.to(device)
                target = target.to(device)
                prd_logits = net(img)
                prd_logits /= linear_temperature
                loss = loss_criterion(prd_logits, target)

                if is_train:
                    train_optimizer.zero_grad()
                    loss.backward()
                    train_optimizer.step()

                total_num += img.shape[0]
                total_loss += loss.item() * data_loader.batch_size
                prediction = torch.argsort(prd_logits, dim=-1, descending=True)
                total_correct_1 += torch.sum((prd_logits.argmax(dim=1) == target).float()).item()
                total_correct_5 += torch.sum(
                    (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            print('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                  .format('Train' if is_train else 'Test', i, num_epo, total_loss / total_num,
                          total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


class ClassificationNetwork(nn.Module):
    def __init__(self, in_model, outdim, linear_protocol=True):
        super(ClassificationNetwork, self).__init__()
        self.network = in_model
        self.network.eval()
        test_input = torch.randn(1, 3, 32, 32).to(device)
        test_out = self.network(test_input).view(1, -1)  # flatten
        print('linear head dim', test_out.shape[-1])
        self.classification_head = nn.Linear(test_out.shape[-1], outdim)
        self.linear_protocol = linear_protocol

    def forward(self, x):
        if self.linear_protocol:
            with torch.no_grad():
                out_features = self.network(x)
        else:
            out_features = self.network(x)  # finetune everything

        out_features = out_features.view(x.shape[0], -1)
        logits = self.classification_head(out_features)
        return logits


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def linear_finetune(num_epochs, net, data_loader, train_optimizer, args, train=True, linear_temperature=0.1,
                    logging=None):
    is_train = train
    net.eval()
    net = net.to(device)
    loss_criterion = nn.CrossEntropyLoss()

    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    results = {'avg_loss': [], 'acc1': [], 'acc5': []}
    for i in range(1, num_epochs + 1):
        if is_train:
            adjust_learning_rate(train_optimizer, i, args)
        with (torch.enable_grad() if is_train else torch.no_grad()):
            for sample in data_loader:
                img, target = sample
                target = target.squeeze()
                img = img.to(device)
                target = target.to(device)
                prd_logits = net(img)
                prd_logits /= linear_temperature
                loss = loss_criterion(prd_logits, target)

                if is_train:
                    train_optimizer.zero_grad()
                    loss.backward()
                    train_optimizer.step()

                total_num += img.shape[0]
                total_loss += loss.item() * data_loader.batch_size
                prediction = torch.argsort(prd_logits, dim=-1, descending=True)
                total_correct_1 += torch.sum((prd_logits.argmax(dim=1) == target).float()).item()
                total_correct_5 += torch.sum(
                    (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            avg_loss = total_loss / total_num
            acc1 = total_correct_1 / total_num * 100
            acc5 = total_correct_5 / total_num * 100
            print('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                  .format('Train' if is_train else 'Test', i, num_epochs, avg_loss,
                          acc1, acc5))
            results['avg_loss'].append(avg_loss)
            results['acc1'].append(acc1)
            results['acc5'].append(acc5)

        if logging is not None:
            data_frame = pd.DataFrame(data=results, index=range(0, i + 1))
            data_frame.to_csv(args.results_dir + '/{}.csv'.format(logging), index_label='epoch')

        torch.save(net.state_dict(), './linear_evaluation.pth')
    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':

    torch.manual_seed(1234)
    np.random.seed(1234)

    train_dir = 'camelyonpatch_level_2_split_train_x.h5'
    train_meta = 'camelyonpatch_level_2_split_train_meta.csv'
    test_dir = 'camelyonpatch_level_2_split_valid_x.h5'
    test_meta = 'camelyonpatch_level_2_split_valid_meta.csv'

    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

    parser.add_argument('-a', '--arch', default='resnet50')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', default=[120, 180, 250], nargs='*', type=int, help='mile stones for fix lr decay')
    parser.add_argument('--cos', default=False, help='use cosine lr schedule')
    parser.add_argument('--warmup_epochs', default=5, help='warm up for cosine schedule')

    parser.add_argument('--batch_size', default=1600, type=int, metavar='N', help='batch size per gpu')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--local_rank', default=0, type=int, help='master rank for ddp')
    parser.add_argument('--enable_parallel', default=True, type=bool, help='enable ddp')

    # moco specific configs:
    parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco_k', default=16000, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco_t', default=0.07, type=float, help='softmax temperature')
    parser.add_argument('--aug_plus', default=True, type=bool, help='MoCo v2 aug_plus')
    parser.add_argument('--bn_splits', default=4, type=int,
                        help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
    parser.add_argument('--num-classes', default=2, type=int)
    parser.add_argument('--symmetric', default=True,
                        help='use a symmetric loss function that backprops to both crops')
    parser.add_argument('--mlp', default=True, help='mlp head')

    # knn monitor
    parser.add_argument('--knn_k', default=255, type=int, help='k in kNN monitor')
    parser.add_argument('--knn_t', default=0.1, type=float,
                        help='softmax temperature in kNN monitor; could be different with moco-t')

    # utils
    # TODO Change path
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='./results_v4/', type=str,
                        help='path to cache (default: none)')

    args = parser.parse_args()  # running in command line
    # get local_rank externally
    local_rank = args.local_rank

    args.cos = True
    args.symmetric = False

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    local_size = torch.cuda.device_count()

    if args.aug_plus:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation([0, 90]),
                # transforms.ColorJitter(0.2, contrast=(0.8, 1.0), saturation=(1.0, 1.0), hue=(0.0, 0.0))
            ], p=0.5),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])]),
            transforms.ToTensor(),
            transforms.Normalize((178.7028, 136.7650, 176.1714), (59.4574, 70.1370, 53.8638))]
        )
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation([0, 90]),
                # transforms.ColorJitter(0.2, contrast=(0.8, 1.0), saturation=(1.0, 1.0), hue=(0.0, 0.0))
            ], p=0.5),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])]),
            transforms.ToTensor(),
            transforms.Normalize((178.7028, 136.7650, 176.1714), (59.4574, 70.1370, 53.8638))]
        )

    negative_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((32, 32)),
        transforms.RandomResizedCrop((32, 32), scale=(1.2, 2), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation([0, 90]),
            # transforms.ColorJitter(0.2, contrast=(0.8, 1.0), saturation=(1.0, 1.0), hue=(0.0, 0.0))
        ], p=0.5),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])]),
        transforms.ToTensor(),
        transforms.Normalize((178.7028, 136.7650, 176.1714), (59.4574, 70.1370, 53.8638))]
    )

    test_transform = transforms.Compose([
        transforms.CenterCrop((32, 32)),
        transforms.Normalize((178.7028, 136.7650, 176.1714), (59.4574, 70.1370, 53.8638))]
    )

    train_h5 = h5py.File(train_dir, 'r')
    test_h5 = h5py.File(test_dir, 'r')
    train_meta = pd.read_csv(train_meta)
    test_meta = pd.read_csv(test_meta)

    train_labels = train_meta.loc[:, ['center_tumor_patch']].to_numpy()
    test_labels = test_meta.loc[:, ['center_tumor_patch']].to_numpy()

    train_dst = BasicDataset(train_h5, train_labels, transform=[train_transform, negative_transform], train=True)
    train_sampler = DistributedSampler(train_dst)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=8, pin_memory=True, drop_last=True)

    valid_dst = BasicDataset(train_h5, train_labels, transform=test_transform, train=False)
    valid_sampler = DistributedSampler(train_dst)
    valid_loader = DataLoader(valid_dst, batch_size=args.batch_size, sampler=valid_sampler, shuffle=False,
                              num_workers=8, pin_memory=True, drop_last=True)

    test_dst = BasicDataset(test_h5, test_labels, test_transform, train=False)
    # test_dst = Subset(test_dst, range(len(test_dst)))
    test_loader = DataLoader(test_dst, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             drop_last=True)

    model = ModelMoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        arch=args.arch,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
        mlp=args.mlp
    ).cuda()
    # Use SyncBN(all_gather)
    # Sharing a BN may leak information
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    if args.enable_parallel:
        # DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # load model if resume
    epoch_start = 1
    if args.resume is not '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        if epoch % 5 == 0:
            test_acc_1 = test(model.module.encoder_q, valid_loader, test_loader, epoch, args)
            results['test_acc@1'].append(test_acc_1)
        else:
            results['test_acc@1'].append('')

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + '/log_moco_path_v4.csv', index_label='epoch')

        # save model
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                   args.results_dir + '/triplet_path_v4.pth')

    print('-------------- Start Finetuning for 100 epochs (linear evaluation protocol) --------')
    linear_eva_model = ClassificationNetwork(model.encoder_k.net[:8], outdim=2)
    linear_eva_model = DDP(linear_eva_model, device_ids=[local_rank], output_device=local_rank)

    eva_optimizer = torch.optim.SGD(linear_eva_model.parameters(), lr=0.1, momentum=0.9)

    loss, top1_acc, top5_acc = linear_finetune(100, linear_eva_model, valid_loader, eva_optimizer, args, train=True,
                                               logging='linear_eva')
    test_loss, test_top1_acc, test_top5_acc = linear_finetune(1, linear_eva_model, test_loader, None, train=False)

    print('-------------- Start Finetuning for 100 epochs (finetune everything) --------')
    finetune_model = ClassificationNetwork(model.encoder_k.net[:8], outdim=2, linear_protocol=False)
    finetune_model = DDP(finetune_model, device_ids=[local_rank], output_device=local_rank)

    finetune_optimizer = torch.optim.SGD(linear_eva_model.parameters(), lr=0.1, momentum=0.9)

    _, _, _ = linear_finetune(100, linear_eva_model, valid_loader, finetune_optimizer, args, train=True)
    f_test_loss, f_test_top1_acc, f_test_top5_acc = linear_finetune(1, linear_eva_model, test_loader, None, train=False,
                                                                    logging='finetune')
