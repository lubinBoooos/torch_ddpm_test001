import math
import os
import sys
import argparse
import torch
from distrib_utils import init_distributed_mode
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision
from torch.utils import data
from ddpm import DDPM
import torch.distributed as dist
from distrib_utils import is_main_process
from tqdm import tqdm
from torch.optim.swa_utils import SWALR, AveragedModel
from distrib_utils import get_world_size

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


def train_one_epoch(model, optimizer, data_loader, scheduler, device, epoch, swa_model=None, swa_scheduler=None):
    # 设置模型为训练模式
    model.train()
    # 初始化平均损失为0
    mean_loss = torch.zeros(1).to(device)
    # 清空优化器的梯度
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(enumerate(data_loader), total=len(data_loader))
    else:
        data_loader = enumerate(data_loader)    
    for it, (images,_) in data_loader:
        loss = model(images.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * it + loss.detach()) / (it + 1)
        
        if is_main_process():
            data_loader.set_description(f'Epoch {epoch} lr={scheduler.get_last_lr()[0]}')
            data_loader.set_postfix(loss=mean_loss)
            # EMA update
            swa_model.update_parameters(model)
            swa_scheduler.step()
            
    scheduler.step()  
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
        
    return mean_loss.item()    



def main(args):
    if not torch.cuda.is_available() :
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("/DataDisk/data/MNIST/my_ddpm") is False:
            os.makedirs("/DataDisk/data/MNIST/my_ddpm")

    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST(root='/DataDisk/data/MNIST', train=True, transform=trans, download=True)
        # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(mnist_train)
    train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)
    

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    # 实例化模型
    model = DDPM(1000, 1, 32, 0.9999)
    model = model.to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    # else:
    #     checkpoint_path = os.path.join("./weights", "initial_weights.pt")
    #     # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    #     if rank == 0:
    #         torch.save(model.state_dict(), checkpoint_path)

    #     dist.barrier()
    #     # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #schd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=60)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000, 2000, 3000])

    swa_scheduler = None
    swa_model = None
    if is_main_process():
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    for epoch in range(4000):
        train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    scheduler=scheduler,
                                    device=device,
                                    epoch=epoch,
                                    swa_model=swa_model,
                                    swa_scheduler=swa_scheduler
                                    )
        if rank == 0:
            tags = ["loss"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            if epoch%1000 == 0:
                save_name = "{}.pth".format(epoch)
                torch.save(swa_model.state_dict(), os.path.join("/DataDisk/data/MNIST/my_ddpm", save_name))

    dist.destroy_process_group()
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--num_classes', type=int, default=10)
    #parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    #parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str, default="/home/wz/data_set/flower_data/flower_photos")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='./weights/initial_weights.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)