#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class CustomMNISTDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        return 60_000

    def __getitem__(self, index):
        x = np.random.randint(0, 255, size=(1, 28, 28))
        x = torch.tensor(x).to(torch.float32)
        # y = torch.tensor(y).to(torch.float32)
        y = torch.ones(1, dtype=torch.long).random_(4)
        return x, y


class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partitions a dataset into different chuncks."""

    # 先对index进行shuffle
    # 然后按照size进行partition
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """Network architecture."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset():
    """Partitioning MNIST"""
    # dataset = datasets.MNIST(
    #     "data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #     ),
    # )
    dataset = CustomMNISTDataset()
    size = int(dist.get_world_size())  # 获取rank的个数
    total_bach_size = 128
    bsz = int(total_bach_size / float(size))  # 每个rank对应的batch size
    partition_sizes = [1.0 / size for _ in range(size)]  # 设置每个rank处理数据量的大小
    partition = DataPartitioner(dataset, partition_sizes)  # 数据切分
    partition = partition.use(dist.get_rank())  # 获取当前rank对应的数据

    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """Gradient averaging."""
    # size = float(dist.get_world_size())
    # for param in model.parameters():
    #     dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
    #     param.grad.data /= size
    # size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.AVG)
        # param.grad.data /= size


def run(rank, size):
    """Distributed Synchronous SGD Example"""
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
    model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in tqdm(train_set, desc=f"epoch: {epoch}, rankid: {rank}"):
            data, target = Variable(data), Variable(target)
            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target.flatten())
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        # print(
        #     "Rank ", dist.get_rank(), ", epoch ", epoch, ": ", epoch_loss / num_batches
        # )


def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    run(rank=rank, size=world_size)


if __name__ == "__main__":
    main()
