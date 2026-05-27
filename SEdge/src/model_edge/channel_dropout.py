import torch

if not hasattr(torch.nn, 'Dropout1d'):
    torch.nn.Dropout1d = torch.nn.Dropout2d


class ChannelDropout(torch.nn.Dropout1d):
    def __init__(self, p = 0.5, inplace = False):
        super().__init__(p, inplace)
    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        return super().forward(x.permute(0,2,1)).permute(0,2,1)