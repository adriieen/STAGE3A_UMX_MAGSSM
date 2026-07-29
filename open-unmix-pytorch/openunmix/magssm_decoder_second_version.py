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

def extend_hermitian_spectrogram(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    """
    Étend un spectrogramme complexe de forme (B, F, T) vers (B, n_fft, T)
    par symétrie hermitienne.
    
    Args:
        x: Tenseur complexe (B, F, T) où F = n_fft // 2 + 1
        n_fft: Taille de la FFT
    Returns:
        x_ext: Tenseur complexe (B, n_fft, T)
    """
    B, F, T = x.shape
    assert F == n_fft // 2 + 1, f"F ({F}) doit valoir n_fft // 2 + 1 ({n_fft // 2 + 1})"
    
    x_ext = torch.zeros(B, n_fft, T, dtype=x.dtype, device=x.device)
    
    # 1. Fréquences positives (DC -> Nyquist)
    x_ext[:, :F, :] = x
    
    # 2. Fréquences négatives (Symétrie miroir complexes conjuguées)
    # k = 1..F-2 s'inversent vers n_fft - 1 .. F
    x_neg = torch.conj(torch.flip(x[:, 1:F-1, :], dims=(1,)))
    x_ext[:, F:, :] = x_neg
    
    return x_ext

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

def perform_overlapp_add(chunks: torch.Tensor, window: torch.Tensor, n_fft: int, n_hop: int,length: Optional[int] = None):
    """Perform overlapp add on a Tensor representing the stacked temporal information to obtain the full signal.
    (B, T, n_fft) --> (B, output_length) where output_length = (T-1)*n_hop + n_fft 
    
    if a length is given, the output is truncated to this length.

    a   b   c
    a   b   c
    a   b   c
    a   b   c
    a   b   c
    a   b   c


    -->

    With a scaling on the sequence that depends on the window used.
        a a a a a a 
    +        b b b b b b 
    +             c c c c c c 
    --------------------------
    retrieved signal: with a scaling performed 
    """

    B, T, L = chunks.shape
    assert L == n_fft


    if window is not None:
        window = window.to(chunks.device)
        chunks = chunks * window.unsqueeze(0).unsqueeze(0)
        
    # 2. Reshape pour F.fold : (B, n_fft, T)
    chunks_fold = chunks.transpose(1, 2)
    output_width = (T - 1) * n_hop + n_fft
    
    # 3. Overlap-Add 
    y_folded = F.fold(
        chunks_fold,
        output_size=(1, output_width),
        kernel_size=(1, n_fft),
        stride=(1, n_hop)
    ).squeeze(1).squeeze(1) # (B, output_width)
    
    # 4. Découpage à la longueur désirée
    if length is not None:
        y_folded = y_folded[:, -length:]
        
    return y_folded


class MagSSM_Decoder(nn.Module): 
    """Trainable decoder that maps a complex spectrogram of shape (B, F, T) to a tensor of waveforms of shape (B, N_0), where N_0 corresponds to the argument "length" of the torch.istft function.
    
    A symetrization of the tensor gives us an input for the SSM of shape (B, n_fft = 2F-2, T).
    The SSM does not change the dimension of the Tensor, but aims at modifying, thanks to it C matrix, the information from frequency domain to time domain
    Finally, overlapp add is performed to retrieve the full signal.
    
    Args :
        length(int): Desired output length.
        n_fft (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        n_hop (int) : parameter of the stft used as the encoder that we want to replicate in a trainable way
        dim_state (int) : dimension of hidden state of the SSM used for the transformation
        window (Tensor, optional): Synthesis/analysis window for COLA normalization buffer calculation.
    """

    def __init__(
        self,
        samplerate: int = 14700,
        n_fft: int = 340,
        n_hop: int = 34,
        length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        dim_state: int = 129,
        og=False,
        B_C_init=None,
        C_C_init="istft",
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

   
    ):
        super(MagSSM_Decoder, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.length = length
        self.window = window if window is not None else torch.hann_window(n_fft)
        self.device = device
        self.mimo = MIMOSSM( 
            d_in= n_fft,
            d_state=dim_state,
            d_out=n_fft,
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
            target_tau = 1,
            effective_samplerate = samplerate/n_hop
        ).to(device)

        self._s_ola_cache = {}

    def forward(self, x: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:

        assert self.window is not None, "The window has to be precised"

        B, F, T = x.data.shape # expects B, F, T

        x = extend_hermitian_spectrogram(x, self.n_fft) # B, N_fft, T

        x = x.transpose(1,2) # B, T = sequence axis, N_fft
        x = self.mimo(x) # B, T, N_fft

        y = perform_overlapp_add(x, self.window, self.n_fft, self.n_hop, length)


        # COLA Normalization if window is provided (cached per length/device/dtype)
        if self.window is not None:
            out_len = y.shape[-1]
            dtype = y.dtype
            cache_key = (out_len, y.device, dtype)
            if cache_key not in self._s_ola_cache:
                s_ola_vec = compute_s_ola(self.window, out_len, self.n_hop).to(device=y.device, dtype=dtype)
                s_ola_vec = torch.where(s_ola_vec == 0, torch.ones_like(s_ola_vec), s_ola_vec)
                self._s_ola_cache[cache_key] = s_ola_vec

            s_ola_vec = self._s_ola_cache[cache_key]
            y = y / s_ola_vec.unsqueeze(0)

        return y


