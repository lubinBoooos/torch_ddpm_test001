import torch
import json

if __name__ == "__main__":

    # prj = torch.nn.Linear(1, 1, bias=False)
    # torch.nn.init.constant_(prj.weight.data, 1.0) 
    # input =  torch.tensor([4.2]).expand(2,-1)
    # output = torch.tensor([4.0]).expand(2,-1)
    
    # r = prj(input)
    # loss = torch.norm(r - output)
    # loss.backward()
    # grad_value1 = prj.weight.grad
    # print(grad_value1)
    
    # print("------------------------------------------")
    
    # prj2 = torch.nn.Linear(1, 1, bias=False)
    # torch.nn.init.constant_(prj2.weight.data, 1.0) 
    # input2 =  torch.tensor([4.2]).expand(2,-1)
    # output2 = torch.tensor([4.0]).expand(2,-1)
    
    # loss2 = torch.nn.functional.mse_loss(prj2(input2), output2)
    # print(loss2)
    
    # loss2.backward()
    # grad_value2 = prj2.weight.grad
    # print(grad_value2)
    
    with open('/DataDisk2/LLMs/MagicDrive/SDv1.5mv-rawbox_2023-09-07_18-39_224x400/controlnet/config.json','r') as fp:
        data = json.load(fp)
        
    print(len(data))    
    print(data.keys())    