##################################################################################################
# 在我看来 DDPM需要有以下的模块
# 1. 生成\alpha, \beta, \hat{\alpha_t}, \hat{\beta_t}
# 2. t ~ U(1, T), \rand ~ N(0, I), x_0 ~ Input
# 3. 计算误差，训练Unet的参数，||\rand - Unet(\hat{\alpha_t}*x_0 + \hat{\beta_t}*\rand,  t)||^2
# 4. Unet 自回归生成, x_{t-1} = \frac{1}{\alpha_t}(x_t - \beta_t*Unet(x_t, t)) + \beta_t*z; z ~ N(0, I)
# 5. 其中的x_t = \hat{\alpha_t}x0 + \sqrt(1-\hat{\alpha_t}^2) \rand
###  我这里的x_t 并没有直接从N(0,I)获得，而是采用趋向于0的\hat{\alpha_t}对噪声加上一些信息
###  难点：（1）Unet中的关于序列顺序t的建模（2）生成超参



#### 难点一，注入t建模的位置编码

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import   os
from tqdm import tqdm
from unet import UNet
from torch.nn import functional as F
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


#### 产生一系列的超参数 
@torch.no_grad()    
def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)    
    
@torch.no_grad()
def generate_hydra_params2(T, is_sincos_hydra=False): 
    if is_sincos_hydra:
        betas = generate_cosine_schedule(T) #
    else:    
        low = 1e-4 * 1000 / T
        high = 0.02 * 1000 / T
        betas = np.linspace(low, high, T)
    
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas) 
    
    hat_alphas = np.sqrt(alphas_cumprod) #\sqrt{\alphas} \sqrt{a1*a2*a3...*aT}
    hat_betas = np.sqrt(1.0 - alphas_cumprod) # \sqrt(1 - a1*a2*...*aT)
    
    result = (alphas, betas, hat_alphas, hat_betas)
    result = [torch.tensor(r, dtype=torch.float32) for r in result]
    return result    
    
  
def cycle(dl):
    while True:
        for data in dl:
            yield data  
    

from copy import deepcopy
class DDPM(torch.nn.Module):
    def __init__(self, T, in_channels, embeding_dim, ema_decay=0.9999, ema_update_rate=1, ema_start=2000, is_sincos_hydra=False, time_emb_dim=128) -> None:
        super().__init__()
        # \eta
        self.unet = UNet(img_channels=1, 
                          base_channels=128, 
                          channel_mults=(1,2,2), 
                          time_emb_dim=time_emb_dim, 
                          norm="gn", 
                          dropout=0.1, 
                          activation=F.silu, 
                          attention_resolutions=(1,))
        #self.alpha,self.beta,self.hat_alpha,self.hat_beta = generate_hydra_params(T)
        self.alpha,self.beta,self.hat_alpha,self.hat_beta = generate_hydra_params2(T, is_sincos_hydra)
        self.T = T
        self.ema_update_rate = ema_update_rate
        self.ema_start = ema_start
        self.ema_decay = ema_decay
        self.ema_unet = deepcopy(self.unet)
    
    #  前向训练Unet的参数   
    def forward(self, x):
        b,c,h,w = x.shape
        x0 = x
        t = torch.randint(0, self.T, (b,))
        # 这里在实现的时候，因为x_t = \alpha_t*x_t-1 +\beta_t*eta 
        # eta = torch.rand(x.shape).to(x.device) # 错在这里，这里是均匀分布，应该用正态分布
        eta = torch.randn(x.shape).to(x.device)
        hat_alpha_t = self.hat_alpha[t].to(x.device).float()
        hat_beta_t = self.hat_beta[t].to(x.device).float()
        t = t.to(x.device)
        x_t = hat_alpha_t[:, None, None, None]*x0 + hat_beta_t[:, None, None, None]*eta
        
        eta_theta = self.unet(x_t, t)
        #r = torch.norm(eta - eta_theta)
        r = F.mse_loss(eta_theta, eta)
        return r
            
    # EMA 不参与更新        
    def ema_update(self, iter, flag=True):
        if iter % self.ema_update_rate == 0:
            if iter < self.ema_start:
                self.ema_unet.load_state_dict(self.unet.state_dict())
            else:
                for current_params, ema_params in zip(self.unet.parameters(), self.ema_unet.parameters()):
                    old, new = ema_params.data, current_params.data
                    ema_params.data = old * self.ema_decay + (1 - self.ema_decay) * new
        
            
            
    #  自回归T步后从噪声生成图像
    @torch.no_grad()
    def sample(self, device):
        # x_t-1 = \frac{1}{\alpha_t}(x_t - \beta_t*eta_theta(x_t, t)) + \beta_t*z
        shape = (1, 1, 28, 28)
        # x = x_T ~ N(0, I)  
        # x = torch.rand(shape).to(device)
        x = torch.randn(shape).to(device)
        with torch.no_grad():
            for t in tqdm(range(0, self.T)):
                #z = torch.rand(shape).to(device)   这是均匀分布
                z = torch.randn(shape).to(device)   #这是正态分布
                t = self.T - t - 1 # 从x_T -> x_0
                
                alpha_t = self.alpha[t].to(device)
                beta_t = self.beta[t].to(device)
                hat_beta_t = self.hat_beta[t].to(device)
                if t == 0:
                    hat_beta_t_1 = torch.tensor([1.0]).to(device)
                else:
                    hat_beta_t_1 = self.hat_beta[t-1].to(device)   
                
                # 这里最后x0一定注意不能加随机噪声了，我当时怎么也出不了清晰的结果，就是这里出问题    
                if t > 0:     
                    t = torch.tensor([t]).to(device)
                    #x = 1.0/alpha_t * (x - (beta_t**2/hat_beta_t)*self.unet(x, t)) + beta_t*z  // sujianlin version
                    x = 1.0/torch.sqrt(alpha_t) * (x - (beta_t/hat_beta_t)*self.unet(x, t)) + torch.sqrt(beta_t)*(hat_beta_t_1/hat_beta_t)*z
                else:
                    t = torch.tensor([t]).to(device)
                    x = 1.0/torch.sqrt(alpha_t) * (x - (beta_t/hat_beta_t)*self.unet(x, t))    
                    
                del alpha_t
                del beta_t
                del t
                del z
                del hat_beta_t
                torch.cuda.empty_cache()
        return x    
        

from distrib_utils import is_main_process, reduce_value
import sys
from torch.optim.swa_utils import SWALR, AveragedModel

def train_by_epoch():
    #### 定义优化器
    model = DDPM(1000, 1, 32, 0.9999, is_sincos_hydra=True) # T  =1000, inchn=1, hidden_chn=32  
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    #schd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=60)
    schd = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1000, 1500])
    
    # EMA
    swa_model = AveragedModel(model)  
    #swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    swa_start = 160
    
    device = torch.device("cuda", 4)  
    model = model.to(device)    
    
    #trans = transforms.ToTensor() 
    # 这里需要将范围卡在[-1,1]而不是[0,1]
    trans = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda sample: 2*sample - 1.0)
    ])
    
    # 不用fashionMNIST 不好分清楚生成的是不是有用的
    # mnist_train = torchvision.datasets.FashionMNIST(root='/DataDisk/data/fashionMNIST', train=True, transform=trans, download=True)
    # data_loder = data.DataLoader(mnist_train, batch_size=256, shuffle=True)
    mnist_train = torchvision.datasets.MNIST(root='/DataDisk/data/MNIST', train=True, transform=trans, download=True)
    data_loder = data.DataLoader(mnist_train, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
    
    for epoch in range(2000):
        train_loss = 0.0
        progress_bar = tqdm(enumerate(data_loder), total=len(data_loder))
        for it, (images, _) in progress_bar:
            loss = model(images.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            progress_bar.set_description(f'Epoch {epoch} lr={schd.get_last_lr()[0]}')
            progress_bar.set_postfix(loss=train_loss / (it + 1))
            
        if epoch > swa_start:
            swa_model.update_parameters(model)
            #swa_scheduler.step()
        
        schd.step()    
        if epoch%200 == 0:
            save_name = "myself_{}.pth".format(epoch)
            save_ema_name = "myself_ema_{}.pth".format(epoch)
            torch.save(model.state_dict(), os.path.join("/DataDisk/data/MNIST/my_ddpm", save_name))
            torch.save(swa_model.state_dict(), os.path.join("/DataDisk/data/MNIST/my_ddpm", save_ema_name))
    
    progress_bar.close()        

def train_by_iteration():
    model = DDPM(1000, 1, 32, 0.9999, time_emb_dim=128*4) # T  =1000, inchn=1, hidden_chn=32    
    model.load_state_dict(torch.load(os.path.join("/DataDisk/data/MNIST/my_ddpm", "my_ddpm_iter100000.pth")))
    
    device = torch.device("cuda", 4)    
    model = model.to(device) 
    
    trans = transforms.ToTensor()
    # mnist_train = torchvision.datasets.FashionMNIST(root='/DataDisk/data/fashionMNIST', train=True, transform=trans, download=True)
    # data_loder = data.DataLoader(mnist_train, batch_size=256, shuffle=True)
    mnist_train = torchvision.datasets.MNIST(root='/DataDisk/data/MNIST', train=True, transform=trans, download=True)
    data_loder = data.DataLoader(mnist_train, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
    
    mnist_loader = cycle(data_loder)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    progress_bar = tqdm(range(1, 8000001), total=8000000)
    train_loss = 0.0
    avg_loss = -1.0
    for iter in progress_bar:
        
        images,_ = next(mnist_loader)
        loss = model(images.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()    
        model.ema_update(iter)

        progress_bar.set_description(f'Iteration {iter} lr={optimizer.param_groups[0]["lr"]}') 
        progress_bar.set_postfix(loss=loss.detach().item(), ema_loss=avg_loss)
        
        if iter % 10 == 0:
            avg_loss = train_loss/10.0
            train_loss = 0.0

        if iter%100000 == 0:
            save_name = "my_ddpm_iter{}.pth".format(iter)
            torch.save(model.state_dict(), os.path.join("/DataDisk/data/MNIST/my_ddpm", save_name))
            

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
def paralled_train_by_epoch(rank, world_size):
    dist.init_process_group("nccl", init_method='tcp://127.0.0.1:29500', rank=rank, world_size=world_size)
    model = DDPM(1000, 1, 32, 0.9999).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    model = model.to(rank)
    
    trans = transforms.ToTensor()
    # mnist_train = torchvision.datasets.FashionMNIST(root='/DataDisk/data/fashionMNIST', train=True, transform=trans, download=True)
    # data_loder = data.DataLoader(mnist_train, batch_size=256, shuffle=True)
    mnist_train = torchvision.datasets.MNIST(root='/DataDisk/data/MNIST', train=True, transform=trans, download=True)
    data_loder = data.DataLoader(mnist_train, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    #schd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=60)
    schd = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4000, 6000])
    for epoch in range(8001):
        if torch.cuda.current_device() == 0:
            train_loss = 0.0
            progress_bar = tqdm(enumerate(data_loder), total=len(data_loder))
        for it, (images, _) in progress_bar:
            loss = model(images.to('cuda:0'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            progress_bar.set_description(f'Epoch {epoch} lr={schd.get_last_lr()[0]}')
            progress_bar.set_postfix(loss=train_loss / (it + 1))
            #ema update
            model.ema_update()
        schd.step()    
        if epoch%1000 == 0:
            save_name = "{}.pth".format(epoch)
            torch.save(model.state_dict(), os.path.join("/DataDisk/data/MNIST/my_ddpm", save_name))


TRAIN = False   
TEST = False
 
if __name__ == "__main__":
    
    if TEST:
        trans = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda sample: 2*sample - 1.0)
        ])
        # mnist_train = torchvision.datasets.FashionMNIST(root='/DataDisk/data/fashionMNIST', train=True, transform=trans, download=True)
        # data_loder = data.DataLoader(mnist_train, batch_size=256, shuffle=True)
        mnist_train = torchvision.datasets.MNIST(root='/DataDisk/data/MNIST', train=True, transform=trans, download=True)
        data_loder = data.DataLoader(mnist_train, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
        image, lable = next(iter(data_loder))
        sample = image[20,0,:,:].numpy()
        
    else:
        if TRAIN:
            print("========================>> ddpm train >>========================")
            #train_by_epoch()
            train_by_iteration()
     
        else:
            print("========================>> ddpm generate image >>========================")
            sin_cos = True
            
            if sin_cos:
                model_dict = torch.load(os.path.join("/DataDisk/data/MNIST/my_ddpm", "myself_1600.pth"))
                model = DDPM(1000, 1, 32, 0.9999, is_sincos_hydra=True, time_emb_dim=128)
            else:
                model_dict = torch.load(os.path.join("/DataDisk/data/MNIST/my_ddpm", "my_ddpm_iter700000.pth"))
                model = DDPM(1000, 1, 32, 0.9999, is_sincos_hydra=False, time_emb_dim=128*4)    
            
            model.load_state_dict(model_dict)
            model = model.to("cuda:1")
            model = model.eval()
            res = model.sample("cuda:1")
            
            res = (res / 2 + 0.5).clamp(0,1).squeeze()
            image = (res*255).round().to(torch.uint8).cpu().numpy()

            #res = res.squeeze(0).squeeze(0).detach().cpu().numpy()
            plt.figure("ddpm_res")
            plt.imshow(image,  interpolation='nearest')
            plt.axis('off')
            plt.savefig("./ddpm_res8.png", bbox_inches='tight', pad_inches=0)
        
    
    