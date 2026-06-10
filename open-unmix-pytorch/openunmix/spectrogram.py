from typing import Optional, Mapping
import torch
from torch import Tensor
import torch.nn as nn
from magssm import MagSSM_Encoder
from path_config import setup_paths, amp_autocast
setup_paths()

class Trainable_spectrogram(nn.Module):

    def __init__(
        self,
        nb_bins: int = 2049,
        nb_channels: int = 2,
        n_hop = 1024,
        dim_state = 129,
        encoder : Optional[nn.Module] = None,
        device = None,
        chunk_duration : Optional[int] = None,
        log_distributed_frequencies= False,
        # conv_downsample_factor: int = 8,   # facteur de downsampling temporel via Conv2D
        
    ):
        super(Trainable_spectrogram, self).__init__()
        self.encoder = encoder
        self.nb_channels = nb_channels
        # self.conv_downsample_factor = conv_downsample_factor
        # ssm_subsampling = max(1, n_hop // conv_downsample_factor)

        self.magssm_encoder = MagSSM_Encoder(
            d_in = 1,
            dim_state = dim_state,
            d_out = nb_bins,
            device = device,
            log_distributed_frequencies = log_distributed_frequencies,
            chunk_duration = chunk_duration,
            subsampling_factor = n_hop
        ).to(device)

        # self.conv_downsample = nn.Sequential(
        #     # Couche 1 : (B, 1,  T_ssm,   nb_bins) → (B, 8,  T_ssm/2, nb_bins)
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=8,
        #         kernel_size=(7, 7),
        #         stride=(2, 1),        
        #         padding=(3, 3),       
        #     ),
        #     nn.GELU(),
        #     # Couche 2 : (B, 8,  T_ssm/2, nb_bins) → (B, 16, T_ssm/4, nb_bins)
        #     nn.Conv2d(
        #         in_channels=8,
        #         out_channels=16,
        #         kernel_size=(5, 5),
        #         stride=(2, 1),        
        #         padding=(2, 2),       
        #     ),
        #     nn.GELU(),
        #     # Couche 3 : (B, 16, T_ssm/4, nb_bins) → (B, 1,  T_stft,  nb_bins)
        #     nn.Conv2d(
        #         in_channels=16,
        #         out_channels=1,
        #         kernel_size=(3, 3),
        #         stride=(2, 1),        
        #         padding=(1, 1),      
        #     ),
        #     nn.GELU(),
        # )


    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if X is None:
            # Single-GPU: compute STFT internally
            with amp_autocast(enabled=False):
                if self.encoder:
                    X = self.encoder(x.float())
                else:
                    raise ValueError('Encoder should not be none')


        _ , _, _, T = X.data.shape

        # Audio enters the pipeline with format (B, 2, L)
        # print("Spectrogram Module input shape : expects (B,2,L)", x.shape)


        x_left, x_right = x[:,0,:], x[:,1,:]
        x_left, x_right = self.magssm_encoder(x_left), self.magssm_encoder(x_right) #( B, T, d_out ) * 2  


        x = torch.cat((x_left[:,None,...], x_right[:,None, ...]), dim=1) # B, 2, T, d_out

        x = torch.abs(x)

        # print("Spectrogram Module output shape : expects (B,2,T,d_out)", x.shape)

        x = x.permute(0,1,3,2) # B, C, F, T like standard STFT.

        x = x[...,:T]

        nb_samples, nb_channels, nb_bins, nb_frames= x.data.shape

        return(x)