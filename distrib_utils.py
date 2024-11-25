# -*- coding:utf-8 -*-
import math
import os
import sys
import argparse
import torch
from tqdm import tqdm
import torch.optim as optim
from torchvision.datasets import mnist
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def init_distributed_mode(args):
    # 检查环境变量 RANK 和 WORLD_SIZE
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ.get('LOCAL_RANK', 0))  # 使用默认值 0
    # 检查环境变量 SLURM_PROCID
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    # 设置 GPU
    torch.cuda.set_device(args.gpu)

    # 设置通信后端
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL

    # 初始化分布式环境
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def cleanup():
    """
    清理函数，用于销毁进程组。
    """
    dist.destroy_process_group()
    

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    # 检查是否支持分布式环境
    if not dist.is_available():
        return False
    # 检查是否已初始化分布式环境
    if not dist.is_initialized():
        return False
    return True
    
def get_world_size():
    # 检查分布式是否可用并已初始化
    if not is_dist_avail_and_initialized():
        return 1
    # 获取分布式大小
    return dist.get_world_size()

def get_rank():
    # 检查分布式环境是否可用并已初始化
    if not is_dist_avail_and_initialized():
        return 0
    # 获取当前进程的分布式排名
    return dist.get_rank()

def is_main_process():
    """
    判断当前进程是否为主进程
    """
    return get_rank() == 0

def reduce_value(value, average=True):
    # 获取当前进程的数量
    world_size = get_world_size()
    # 如果进程数量小于2，表示单GPU的情况，直接返回value
    if world_size < 2: 
        return value

    # 在不计算梯度的情况下，将value进行所有进程的求和操作
    with torch.no_grad():
        # 使用分布式训练库进行所有进程的求和操作
        dist.all_reduce(value)
        # 如果average为True，则将value除以进程数量，得到平均值
        if average:
            value /= world_size

    return value


# 定义模型
class CNNNet(torch.nn.Module):

    def __init__(self, in_channel, out_channel_one, out_channel_two, out_channel_three, fc_1, fc_2, fc_out):
        super(CNNNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel_one, kernel_size=5, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channel_one, out_channels=out_channel_two, kernel_size=5, stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2,padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=out_channel_two,out_channels=out_channel_three, kernel_size=5, stride=1,padding=1)

        self.fc1 = torch.nn.Linear(5*5*32, fc_1)
        self.fc2 = torch.nn.Linear(fc_1, fc_2)
        self.output = torch.nn.Linear(fc_2, fc_out)

    def forward(self, x):
        x = self.pool1(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool2(torch.nn.functional.relu(self.conv2(x)))
        x = torch.nn.functional.relu(self.conv3(x))

        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.softmax(self.output(x), dim=1)

        return x




def evaluate(model, data_loader, device):
    # 将模型设置为评估模式
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 遍历数据加载器中的每个批次
    for step, data in enumerate(data_loader):
        # 获取图像和标签
        images, labels = data
        # 使用模型进行预测
        pred = model(images.to(device))
        # 获取预测结果中的最大值
        pred = torch.max(pred, dim=1)[1]
        # 统计预测正确的样本个数
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # 将结果进行归一化处理
    sum_num = reduce_value(sum_num, average=False)

    # 返回预测正确的样本个数
    return sum_num.item()




