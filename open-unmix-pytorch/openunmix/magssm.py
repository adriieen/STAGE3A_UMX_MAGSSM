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


from path_config import setup_paths
setup_paths()


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
        log_distributed_frequencies = False

    ):
        
        super(MagSSM, self).__init__()
  
        self.mimo = MIMOSSM( 
            d_in = 1,
            d_state = dim_state,
            d_out = d_out,
            progressive = True,
            chunk_duration = chunk_duration,
            subsampling_factor = subsampling_factor,
            log_distributed_frequencies = log_distributed_frequencies,
            B_C_init='ones',
            C_C_init= None,
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
        x = x[..., None]  #(2B, T, 1)
        x = self.mimo(x) #(2B, T/subsampling_factor, d_out)
        
        _, nb_frames, d_out = x.shape

        x = x.reshape(nb_samples, nb_channels, nb_frames, d_out)
        x = x.permute(0, 1, 3, 2)    # B,C,F,T

        return x


        
class MagSSM_Encoder(nn.Module):
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
        d_in : int = 2,
        dim_state: int = 129,
        d_out: int = 129,
        device = None,
        chunk_duration : Optional[int] = None,
        subsampling_factor : int = 1024,
        log_distributed_frequencies = False
    ):
        
        super(MagSSM_Encoder, self).__init__()
  
        self.mimo = MIMOSSM( 
            d_in = d_in,
            d_state = dim_state,
            d_out = d_out,
            progressive = True,
            chunk_duration = chunk_duration,
            subsampling_factor = subsampling_factor,
            log_distributed_frequencies = log_distributed_frequencies,
            B_C_init='ones',
            C_C_init= None,
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
  
        nb_samples, nb_channels, nb_timesteps = x.data.shape #(B,2,T) - audio

        x = x.permute(0,2,1) #(B,T,2)

        x = self.mimo(x) #(B, T/subsampling_factor, d_out)
        
        return x






        



        