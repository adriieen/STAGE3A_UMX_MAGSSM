import torch
from torch.nn import functional as F

eps_stability = 1e-2
Lambda = torch.tensor([[-1e-4, 1], [-1e-3, 1], [-1e-2, 1], [-1e-1, 1]],dtype= torch.float64)


Lambda.data[:, 0] = -F.relu(- (Lambda.data[:, 0] + eps_stability)) - eps_stability

print(Lambda.data[:,0] > eps_stability)

print(Lambda)