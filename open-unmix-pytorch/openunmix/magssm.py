from typing import Optional, Mapping
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from filtering import wiener
from transforms import make_filterbanks, ComplexNorm
import model


sys.path.append('/home/adubois/openunmix/OpenUnmix/SEdge/src')


from model_edge.mimo_ssm import MIMOSSM


class MagSSM(nn.Module):
    """Trainable spectrogram

    Args :
    
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        n_fft (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        n_hop (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        dim_state (int) : dimension of hidden state of the SSM used for the transformation
    """

    def __init__(
        self,
        dim_state: int = 129,
        d_out: int = 129,
        device = None,
        chunk_duration : Optional[int] = None,
        subsampling_factor : int = 1024,
        mel = False

    ):
        
        super(MagSSM, self).__init__()
  
        self.mimo = MIMOSSM( 
            d_in = 1,
            d_state = dim_state,
            d_out = d_out,
            use_magssm = True,
            chunk_duration = chunk_duration,
            subsampling_factor = subsampling_factor,
            mel = mel
            )
        
        self.device = device
        

    def forward(self, x:Tensor) -> Tensor:

        """Trainable STFT forward path
            Args:
                x (Tensor): audio waveform of
                    shape (nb_samples, nb_channels, nb_timesteps)
            Returns:
                MagsSSM object (Tensor): complex 'stft of a kind' of
                    shape (nb_samples, nb_channels, nb_bins, nb_frames)
                    last axis is stacked real and imaginary
            """
  
        nb_samples, nb_channels, nb_timesteps = x.data.shape #(B,2,T)

        x = x.reshape (nb_samples*nb_channels, nb_timesteps) #(2*B,T)
        
    

   #Unsuccesful test with mix of temporal and channel axes.
    
        # nb_frames = nb_timesteps//self.n_hop                      # nb of temporal frames in the "trainable spectrogram"
            # frames_per_window = self.n_fft // self.n_hop
            # X = torch.zeros((nb_samples, nb_channels, self.nb_bins, nb_frames), dtype=torch.complex64 ).to(x.device)
            # N_windows = nb_timesteps//self.n_fft
            # for n_window in range(N_windows):
            #     y = x[:,n_window*self.n_fft : (1+n_window)*self.n_fft] #(2*B, N_fft)
            #     y = y[...,None]                                         #(2B, N_fft, 1)
            #     y = y.reshape(-1, frames_per_window, self.n_hop) #(2B, N_fft//hop_size, hop_size)
            #     y = self.mimo(y)                                        # (2B, N_fft//hop_size, nb_bins)
            #     y = y.transpose(1,2)
            #     y = y[..., None]
            #     y = y.reshape(nb_samples, nb_channels, self.nb_bins, frames_per_window) #(B, 2, nb_bins, N_fft//hopsize)
            #     X[..., n_window*frames_per_window : (1+n_window)*frames_per_window] += y
            # return X
             



        x = x[..., None]  #(2B, T, 1)
        x = self.mimo(x) #(2B, T/subsampling_factor, d_out)
        
        _, nb_frames, d_out = x.shape

        x = x.reshape(nb_samples, nb_channels, nb_frames, d_out)
        x = x.permute(0, 1, 3, 2)    # B,C,F,T

        return x


        






        



        