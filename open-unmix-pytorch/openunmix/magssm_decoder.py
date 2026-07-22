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


class MagSSM_Decoder(nn.Module): 
    """Trainable decoder that maps a complex spectrogram of shape (B, F, T) to a tensor of waveforms of shape (B, N_0), where N_0 corresponds to the argument "lenght" of the torch.istft function.
    
    The decoder proceeds by padding the input at the left boundary with P = (n_fft-n_hop)/n_hop zero frames, then splitting the frames into groups of M = 2P = 2(n_fft-n_hop)/n_hop.  

    The padding of the left extremity of the signal is for coherence purposes. When performing ISTFT with center=True, the first n_fft samples of the output are obtained from half the amount of spectrogram frames than the others. (since half of it is discarded in the end)
    Since we design the SSM to work with a fixed amount of spectrogram frames, we perform padding on the left extremity frames of the input to ensure that the first n_fft samples (which will still be discarded in the end) will be the result of the processing of a group of frames of 
    the same cardinality as the others.

    The padding on the right extremity enables the last group (i.e. the one containing the last frame) to have the same cardinality than the others.

    Each group gets its frequency axis extended to the value n_fft to replicate the spectrogram in the negative frequencies. The mapping happens like so: X_new[k] = X[k] for k in [0,F] and X_new[nfft - k] = \bar X[k] for k in [1,F-1].
    The frequency vector is then transposed : (n_fft, M) --> (M, n_fft) and mapped to a unidimensional waveform of size (1, n_fft).

    This process is done with a stride s = L/n_hop and the resulting waveforms are then concatenated across the time dimension. 

    The first n_fft samples of the concatenated signal are disarded. Indeed, the additionnal padded frames contribute for n_fft/2 samples in the final signal, which we have to add to the n_fft/2 samples that have to be discarded when 
    computing the torch.istft method with center = True. Consequently, this method suppose that the time-frequency representation of the signal we analyse has been obtained with an equivalent of the "center = True" argument of the STFT.

    Finally, we limit ourselves to the N_0 first samples of the obtained time sequence to match the desired length. 
    
    Args :
        length(int): Desired output length.
        n_fft (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        n_hop (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        dim_state (int) : dimension of hidden state of the SSM used for the transformation
    """

    def __init__(
        self,
        n_fft: int = 340,
        n_hop: int = 34,
        length : Optional[int] = None,
        dim_state: int = 129,
        og = False,
        B_C_init= None,
        C_C_init= None,
        device = None,
        progressive = None,
        chunk_duration : Optional[int] = None,
        log_distributed_frequencies = False,
        eps_stability: float = 0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        re_lower: float = None,
        re_upper: float = None,
        ensure_stability: str = 'abs',
        sigmoid_scale: float = 1.0,
        structured_initialisation = False
    ):
        
        super(MagSSM_Decoder, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.length = length
        self.single_sequence_decoder = MagSSM_Decoder_one_sequence( 
            d_in = (2*(n_fft - n_hop)//n_hop),
            dim_state = dim_state,
            d_out = 1,
            og = og,
            B_C_init= B_C_init,
            C_C_init= C_C_init,
            device = device,
            progressive = progressive,
            chunk_duration = chunk_duration,
            log_distributed_frequencies = log_distributed_frequencies,
            eps_stability = eps_stability,
            dt_min = dt_min,
            dt_max = dt_max,
            re_lower = re_lower,
            re_upper = re_upper,
            ensure_stability = ensure_stability,
            sigmoid_scale = sigmoid_scale,
            structured_initialisation=structured_initialisation
            )
        
        self.device = device



    def forward(self, x: torch.Tensor) -> torch.Tensor:


        B, F, T = x.data.shape # expects B, F, T
        # print("input dimension", B,F,T)

        group_length = 2*(self.n_fft - self.n_hop)//self.n_hop
        # print("group length", group_length)

        num_padding_frames_left = group_length//2   # so that it contributes to perform nfft//2 samples in the output, 
        # print("num padding frames left", num_padding_frames_left)

        stride = self.n_fft // self.n_hop
        # print("stride", stride)
        
        num_groups = ( (T + num_padding_frames_left - group_length) // stride + 3) # number of groups of frames of length 'group_length' to cover the T frames with stride 'stride'.
        # print("num_groups", num_groups)
        num_padding_frames_right = group_length - (T+num_padding_frames_left - stride*(num_groups-1))  # so that the spectrogram can be divided into groups of even size with stride s.


        #                                           Last group, starting at frame stride * (num_groups -1)
        #                                            ----------------------------  
        #                                            |                          |
        #                                            |     T+n_left             |
        # --------------------------------------------------|-------------------
        # --------------------------------------------------|********************
        #   sequence + padding at left boundary             padding_right

        padding_frames_left = torch.zeros(B, F, num_padding_frames_left).to(self.device)
        padding_frames_right = torch.zeros(B, F, num_padding_frames_right).to(self.device)

        x = torch.cat((padding_frames_left, x, padding_frames_right), dim=-1) # B, F, T' = covers the full sequence with groups of same size with stride s.
        # print("extended_input_dimension", x.data.shape)
        x = x.unfold(dimension=2, size=group_length, step=stride) # B, F, num_groups, group_length
        # print("unfolded input dimension", x.data.shape)
        x = x.transpose(1, 2) # B, num_groups, F, group_length

        x_extended = torch.zeros(B, num_groups, self.n_fft, group_length, dtype=torch.complex64).to(self.device) # F = n_fft //2 +1 

        x_extended[:,:, :F , :] = x
        x_extended[:, :, F:, :] = torch.flip(torch.conj(x[:, :, 1:F-1, :]), dims=(2,))


        y = torch.zeros(B, self.n_fft * num_groups, dtype = torch.complex64).to(self.device)


        for i in range(num_groups):
            output_sequence = self.single_sequence_decoder(x_extended[:,i,:,:]) # (B,n_fft)
            y[:, i*self.n_fft:(i+1)*self.n_fft] += output_sequence

        # print("signal length ; n_fft ; output_shape",self.length, self.n_fft, y.shape)
        return y[:,self.n_fft : self.n_fft+self.length]

        

        
class MagSSM_Decoder_one_sequence(nn.Module):
    """Maps a group of shape (B, N, M) to a waveform of shape (B, N) with a MimoSSM.

    Args :
        N(int): Desired output length.
        B (int): Batch size.
        M (int) : channel dimension (in the global pipeline, equal to the group cardinality)
    """

    def __init__(
        self,
        d_in : int = 18,
        dim_state: int = 129,
        d_out: int = 1,
        og = False,
        B_C_init= None,
        C_C_init= None,
        device = None,
        progressive = False,
        chunk_duration : Optional[int] = None,
        log_distributed_frequencies = False,
        eps_stability: float = 0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        re_lower: float = None,
        re_upper: float = None,
        ensure_stability: str = 'abs',
        sigmoid_scale: float = 1.0,
        structured_initialisation = False
    ):
        
        super(MagSSM_Decoder_one_sequence, self).__init__()
  
        self.mimo = MIMOSSM( 
            d_in = d_in,
            d_state = dim_state,
            d_out = d_out,
            og = og,
            progressive = progressive,
            chunk_duration = chunk_duration,
            log_distributed_frequencies = log_distributed_frequencies,
            B_C_init= B_C_init,
            C_C_init= C_C_init,
            eps_stability = eps_stability,
            dt_min = dt_min,
            dt_max = dt_max,
            re_lower = re_lower,
            re_upper = re_upper,
            stability = ensure_stability,
            sigmoid_scale = sigmoid_scale,
            structured_initialisation=structured_initialisation
            )
        
        self.device = device
        

    def forward(self, x:Tensor) -> Tensor:

        """Trainable decoder forward path
            Args :
        N(int): Desired output length.
        B (int): Batch size.
        M (int) : channel dimension (in the global pipeline, equal to the group cardinality)
            Returns:
                Waveform of size (B,N)
            """
  
        B, sequence_length, channels = x.data.shape #(B,N,M) - transposed set of complex spectrograms frames

        x = self.mimo(x) #(B, N, 1)

        x = x.squeeze(-1) # (B,N)
        # print(x)
        return x






        



        