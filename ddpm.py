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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class sinPosEmbedding(torch.nn.Module):
    '''
    embeds: {channel_dim}
    '''
    def __init__(self, T, channel_dim) -> None:
        super().__init__()
        # {T,1}
        pos = torch.arange(T).unsqueeze(1).float() # 0,1,2,3,...,T-1 
        
        eta = -np.log(10000.0)/channel_dim #
        # e^-\eta*t
        # {1,channel_dim//2}
        e_t = torch.exp(torch.arange(0, channel_dim, 2).unsqueeze(0).float() * eta)
        
        embeddings = torch.zeros(T, channel_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(pos * e_t)
        embeddings[:, 1::2] = torch.cos(pos * e_t)
        
        self.embeddings = embeddings
    #    
    def forward(self, t):
        embeds = self.embeddings[t]
        return embeds
 
class Unet(torch.nn.Module):
    def __init__(self, in_channels, embeding_dim, T) -> None:
        super().__init__() 
        self.in_conv = torch.nn.Conv2d(in_channels, embeding_dim, 3, 1, 1)
        
        # (W-K+2p)/S + 1 = (W-3+1)/2 + 1 = W/2
        
        self.down1 = torch.nn.Conv2d(embeding_dim, embeding_dim, kernel_size=3, stride=2, padding=1) #//2  28//2=14 
        self.down2 = torch.nn.Conv2d(embeding_dim, embeding_dim, kernel_size=3, stride=2, padding=1) #//4  14//2=7
        
        self.mid = torch.nn.Conv2d(embeding_dim, embeding_dim, kernel_size=3, stride=1, padding=1)
        # (W-1)S+K-2P + 1 = (W-1)*2+3-2 + = 2W
        self.up1 = torch.nn.ConvTranspose2d(embeding_dim, embeding_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = torch.nn.ConvTranspose2d(embeding_dim, embeding_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        
        self.out_conv = torch.nn.Conv2d(embeding_dim, in_channels, 3, 1, 1)
        
        # PE
        self.pe = sinPosEmbedding(T, embeding_dim)
        
    def forward(self, x, t):
        x = self.in_conv(x)
        # add PE
        x += self.pe(t)[:, :, None, None].to(x.device)
        x = self.down1(x)   
        x += self.pe(t)[:, :, None, None].to(x.device)
        x = self.down2(x)
        x += self.pe(t)[:, :, None, None].to(x.device)
        x = self.mid(x)
        x += self.pe(t)[:, :, None, None].to(x.device)
        x = self.up1(x)
        x += self.pe(t)[:, :, None, None].to(x.device)
        x = self.up2(x)
        x = self.out_conv(x)
        return x

#### 产生一系列的超参数 
@torch.no_grad()
def generate_hydra_params(T):
        # \sqrt{1-\frac{0.02*t}{T}}
        alpha = np.sqrt(1 - 0.02*np.arange(1, T+1)/T)
        beta = np.sqrt(1 - alpha**2)
        # \hat{alpha} = \alpha1*\alpha2*...\alphaT
        hat_alpha = np.cumprod(alpha)
        hat_beta = np.sqrt(1 - hat_alpha**2)
        result = (alpha, beta, hat_alpha, hat_beta)
        result = [torch.tensor(r) for r in result]
        return result


class DDPM(torch.nn.Module):
    def __init__(self, T, in_channels, embeding_dim) -> None:
        super().__init__()
        # \eta
        #self.noise = Unet(in_channels, embeding_dim, T)
        
        self.noise = UNet(img_channels=1, 
                          base_channels=128, 
                          channel_mults=(1,2,2), 
                          time_emb_dim=128*4, 
                          norm="gn", 
                          dropout=0.1, 
                          activation=F.silu, 
                          attention_resolutions=(1,))
        self.alpha,self.beta,self.hat_alpha,self.hat_beta = generate_hydra_params(T)
        self.T = T
    
    #  前向训练Unet的参数   
    def forward(self, x):
        b,c,h,w = x.shape
        x0 = x
        t = torch.randint(0, self.T, (b,))
        # 这里在实现的时候，因为x_t = \alpha_t*x_t-1 +\beta_t*eta 
        eta = torch.rand(x.shape).to(x.device)
        hat_alpha_t = self.hat_alpha[t].to(x.device).float()
        hat_beta_t = self.hat_beta[t].to(x.device).float()
        t = t.to(x.device)
        x_t = hat_alpha_t[:, None, None, None]*x0 + hat_beta_t[:, None, None, None]*eta
        
        eta_theta = self.noise(x_t, t)
        r = torch.norm(eta - eta_theta)
        return r
            
    #  自回归T步后从噪声生成图像
    def sample(self, device):
        # x_t-1 = \frac{1}{\alpha_t}(x_t - \beta_t*eta_theta(x_t, t)) + \beta_t*z
        shape = (1, 1, 28, 28)
        # x = x_T ~ N(0, I)  
        x = torch.rand(shape).to(device)
        with torch.no_grad():
            for t in tqdm(range(0, self.T)):
                z = torch.rand(shape).to(device)  
                t = self.T - t - 1 # 从x_T -> x_0
                alpha_t = self.alpha[t].to(device)
                beta_t = self.beta[t].to(device)
                t = torch.tensor([t]).to(device)
                x = 1.0/alpha_t * (x - beta_t*self.noise(x, t)) + beta_t*z
                del alpha_t
                del beta_t
                del t
                del z
                torch.cuda.empty_cache()
            
            
        return x    
        

####  这里我们使用一个fashion_mnist来训练，因为unet部分很简单，这里就采用这个数据集来验证下DDPM的正确性
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='/DataDisk/data/fashionMNIST', train=True, transform=trans, download=True)
data_loder = data.DataLoader(mnist_train, batch_size=64, shuffle=True)


#### 定义优化器
model = DDPM(1000, 1, 32) # T  =1000, inchn=1, hidden_chn=32
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
schd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=60)



 
if __name__ == "__main__":
    # test_sinPE = sinPosEmbedding(10, 32)
    # ret = test_sinPE([1,2,3])
    # print(ret.shape)    
    
    # for image, _ in data_loder:
    #     print(image.shape)
    #     break
    
    # image, lable = next(iter(data_loder))
    # print(image.shape)   
    
    # model = model.to('cuda:0')
    
    # for epoch in range(100):
    #     train_loss = 0.0
    #     progress_bar = tqdm(enumerate(data_loder), total=len(data_loder))
    #     for it, (images, _) in progress_bar:
    #         loss = model(images.to('cuda:0'))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         train_loss += loss.detach().item()
    #         progress_bar.set_description(f'Epoch {epoch} lr={schd.get_last_lr()[0]}')
            
    #         progress_bar.set_postfix(loss=train_loss / (it + 1))
            
            
    #     schd.step()    
        
    #     if epoch%10 == 0:
    #         save_name = "{}.pth".format(epoch)
    #         torch.save(model.state_dict(), os.path.join("/DataDisk/data/fashionMNIST/my_ddpm", save_name))
    
    
    model.load_state_dict(torch.load(os.path.join("/DataDisk/data/fashionMNIST/my_ddpm", "90.pth")))
    model = model.cuda()
    model = model.eval()
    res = model.sample("cuda:0")
    print(res.shape)
    
    res = res.squeeze(0).squeeze(0).detach().cpu().numpy()
    
    
    plt.figure("ddpm_res")
    plt.imshow(res,  interpolation='nearest')
    plt.axis('off')
    plt.savefig("./ddpm_res.png", bbox_inches='tight', pad_inches=0)
    
    
    