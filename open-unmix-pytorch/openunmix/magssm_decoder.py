from typing import Optional, Mapping
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from path_config import setup_paths
setup_paths()

from model_edge.mimo_ssm import MIMOSSM


def compute_s_ola(window: torch.Tensor, n_samples: int, n_hop: int) -> torch.Tensor:
    """Computes the exact Constant Overlap-Add (COLA) normalization buffer s_OLA[n]
    for n in [0, n_samples - 1].

    Formula: s_OLA(w)(n) = sum_{m=-infty}^{infty} w^2(n - m * n_hop)
    """
    L = window.shape[0]
    w2 = (window ** 2).to(window.device).float()
    num_frames = (n_samples + L - 1) // n_hop + 1
    output_width = (num_frames - 1) * n_hop + L

    w2_expanded = w2.view(1, L, 1).repeat(1, 1, num_frames)
    s_ola_2d = F.fold(
        w2_expanded,
        output_size=(1, output_width),
        kernel_size=(1, L),
        stride=(1, n_hop)
    )
    s_ola = s_ola_2d.view(-1)[:n_samples]
    return s_ola


class MagSSM_Decoder(nn.Module): 
    """Trainable decoder that maps a complex spectrogram of shape (B, F, T) to a tensor of waveforms of shape (B, N_0), where N_0 corresponds to the argument "length" of the torch.istft function.
    
    The decoder proceeds by padding the input at the left boundary with P = (n_fft-n_hop)/n_hop zero frames, then splitting the frames into groups of M = 2P = 2(n_fft-n_hop)/n_hop.  

    The padding of the left extremity of the signal is for coherence purposes. When performing ISTFT with center=True, the first n_fft samples of the output are obtained from half the amount of spectrogram frames than the others. (since half of it is discarded in the end)
    Since we design the SSM to work with a fixed amount of spectrogram frames, we perform padding on the left extremity frames of the input to ensure that the first n_fft samples (which will still be discarded in the end) will be the result of the processing of a group of frames of 
    the same cardinality as the others.

    The padding on the right extremity enables the last group (i.e. the one containing the last frame) to have the same cardinality than the others.

    Each group gets its frequency axis extended to the value n_fft to replicate the spectrogram in the negative frequencies. The mapping happens like so: X_new[k] = X[k] for k in [0,F] and X_new[nfft - k] = \bar X[k] for k in [1,F-1].
    The frequency vector is then transposed : (n_fft, M) --> (M, n_fft) and mapped to a unidimensional waveform of size (1, n_fft).

    This process is done with a stride s = L/n_hop and the resulting waveforms are then concatenated across the time dimension. 

    The first n_fft samples of the concatenated signal are discarded. Indeed, the additional padded frames contribute for n_fft/2 samples in the final signal, which we have to add to the n_fft/2 samples that have to be discarded when 
    computing the torch.istft method with center = True. Consequently, this method suppose that the time-frequency representation of the signal we analyse has been obtained with an equivalent of the "center = True" argument of the STFT.

    Finally, we limit ourselves to the N_0 first samples of the obtained time sequence to match the desired length. 
    
    Args :
        length(int): Desired output length.
        n_fft (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        n_hop (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        dim_state (int) : dimension of hidden state of the SSM used for the transformation
        window (Tensor, optional): Synthesis/analysis window for COLA normalization buffer calculation.
    """

    def __init__(
        self,
        n_fft: int = 340,
        n_hop: int = 34,
        length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        dim_state: int = 129,
        og=False,
        B_C_init=None,
        C_C_init=None,
        device=None,
        progressive=None,
        chunk_duration: Optional[int] = None,
        log_distributed_frequencies=False,
        eps_stability: float = 0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        re_lower: float = None,
        re_upper: float = None,
        ensure_stability: str = 'abs',
        sigmoid_scale: float = 1.0,
        structured_initialisation=False,
        phase_correction=False,
   
    ):
        super(MagSSM_Decoder, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.length = length
        self.window = window
        self.single_sequence_decoder = MagSSM_Decoder_one_sequence( 
            d_in=(2 * (n_fft - n_hop) // n_hop),
            dim_state=dim_state,
            d_out=1,
            og=og,
            B_C_init=B_C_init,
            C_C_init=C_C_init,
            device=device,
            progressive=progressive,
            chunk_duration=chunk_duration,
            log_distributed_frequencies=log_distributed_frequencies,
            eps_stability=eps_stability,
            dt_min=dt_min,
            dt_max=dt_max,
            re_lower=re_lower,
            re_upper=re_upper,
            ensure_stability=ensure_stability,
            sigmoid_scale=sigmoid_scale,
            structured_initialisation=structured_initialisation,
            phase_correction=phase_correction,
            synthesis_window=window,
            n_hop_for_ola=n_hop
        )
        self.device = device
        self._s_ola_cache = {}

    def forward(self, x: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:

        assert self.window is not None, "The window has to be precised"

        B, F, T = x.data.shape # expects B, F, T

        group_length = 2 * (self.n_fft - self.n_hop) // self.n_hop
        num_padding_frames_left = group_length // 2
        stride = self.n_fft // self.n_hop

        num_groups = ((T + num_padding_frames_left - group_length) // stride + 5)
        num_padding_frames_right = group_length - (T + num_padding_frames_left - stride * (num_groups - 1))

        padding_frames_left = torch.zeros(B, F, num_padding_frames_left, device=x.device, dtype=x.dtype)
        padding_frames_right = torch.zeros(B, F, num_padding_frames_right, device=x.device, dtype=x.dtype)

        x = torch.cat((padding_frames_left, x, padding_frames_right), dim=-1)
        x = x.unfold(dimension=2, size=group_length, step=stride)
        x = x.transpose(1, 2) # B, num_groups, F, group_length

        x_extended = torch.zeros(B, num_groups, self.n_fft, group_length, dtype=x.dtype, device=x.device)
        x_extended[:, :, :F, :] = x
        x_extended[:, :, F:, :] = torch.flip(torch.conj(x[:, :, 1:F-1, :]), dims=(2,))

        # Vectorization across (B * num_groups) for 1-pass parallel GPU execution
        x_batched = x_extended.reshape(B * num_groups, self.n_fft, group_length).contiguous()

        # Single GPU call through single_sequence_decoder (MIMOSSM)
        output_batched = self.single_sequence_decoder(x_batched) # (B * num_groups, n_fft)

        output_groups = output_batched.reshape(B, num_groups, self.n_fft)

        y = torch.zeros(B, self.n_fft * num_groups, dtype=output_groups.dtype, device=x.device)
        for i in range(num_groups):
            y[:, i * self.n_fft : (i + 1) * self.n_fft] += output_groups[:, i, :]

        target_length = length if length is not None else self.length
        out_len = target_length if target_length is not None else (T - 1) * self.n_hop
        
        y_cropped = y[:, self.n_fft : self.n_fft + out_len]

        # COLA Normalization if window is provided (cached per length/device/dtype)
        if self.window is not None:
            dtype = x.real.dtype if x.is_complex() else x.dtype
            cache_key = (out_len, x.device, dtype)
            if cache_key not in self._s_ola_cache:
                s_ola_vec = compute_s_ola(self.window, out_len, self.n_hop).to(device=x.device, dtype=dtype)
                s_ola_vec = torch.where(s_ola_vec == 0, torch.ones_like(s_ola_vec), s_ola_vec)
                self._s_ola_cache[cache_key] = s_ola_vec

            s_ola_vec = self._s_ola_cache[cache_key]
            y_cropped = y_cropped / s_ola_vec.unsqueeze(0)

        return y_cropped


class MagSSM_Decoder_one_sequence(nn.Module):
    """Maps a group of shape (B, N, M) to a waveform of shape (B, N) with a MIMOSSM.

    Args :
        d_in (int) : channel dimension (equal to group cardinality M)
        dim_state (int) : hidden state dimension of SSM
        d_out (int) : output dimension (1 for waveform)
    """

    def __init__(
        self,
        d_in: int = 18,
        dim_state: int = 129,
        d_out: int = 1,
        og=False,
        B_C_init=None,
        C_C_init=None,
        device=None,
        progressive=False,
        chunk_duration: Optional[int] = None,
        log_distributed_frequencies=False,
        eps_stability: float = 0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        re_lower: float = None,
        re_upper: float = None,
        ensure_stability: str = 'abs',
        sigmoid_scale: float = 1.0,
        structured_initialisation=False,
        phase_correction=False,
        synthesis_window=None,
        n_hop_for_ola=None
    ):
        super(MagSSM_Decoder_one_sequence, self).__init__()

        self.mimo = MIMOSSM( 
            d_in=d_in,
            d_state=dim_state,
            d_out=d_out,
            og=og,
            progressive=progressive,
            chunk_duration=chunk_duration,
            log_distributed_frequencies=log_distributed_frequencies,
            B_C_init=B_C_init,
            C_C_init=C_C_init,
            eps_stability=eps_stability,
            dt_min=dt_min,
            dt_max=dt_max,
            re_lower=re_lower,
            re_upper=re_upper,
            stability=ensure_stability,
            sigmoid_scale=sigmoid_scale,
            structured_initialisation=structured_initialisation,
            init_with_positive_frequencies=False,
            domain = "time",
            phase_correction=phase_correction,
            synthesis_window=synthesis_window,
            n_hop_for_ola=n_hop_for_ola
        )
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        x = self.mimo(x) # (B, N, 1)
        x = x.squeeze(-1) # (B, N)
        return x