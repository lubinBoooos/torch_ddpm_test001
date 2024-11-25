import torch



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