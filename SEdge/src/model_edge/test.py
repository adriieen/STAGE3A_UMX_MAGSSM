import torch 
import numpy as np


N = 512 

mel_scale = torch.linspace(0, np.log(1+np.pi), N)
omega = (torch.exp(mel_scale) - 1).to(torch.float32)

target = np.pi/4
i=0

for freq in omega:
    if freq<target:
        i+=1
    
    

print(i)