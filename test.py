import torch

t = torch.randn((2,1))

t1 = t[1]
print(t.shape, t1.shape)