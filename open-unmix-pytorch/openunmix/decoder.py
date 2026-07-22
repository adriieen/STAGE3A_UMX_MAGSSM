from typing import Optional, Mapping
import torch
from torch import Tensor
import torch.nn as nn
from magssm_decoder import MagSSM_Decoder
from path_config import setup_paths, amp_autocast
setup_paths()


class Trainable_decoder(nn.Module):
    """Wrapper for the magssm decoder pipeline, valid for complex spectrograms of shape (B,C,F,T,2) with last dimension containing the real and imaginary parts.
    """

    def __init__(
        self,
        n_fft: int = 340,
        n_hop = 34,
        length: Optional[int] = None,
        return_complex_signal = False,
        dim_state = 129,
        og = False,
        B_C_init = None,
        C_C_init = None,
        device = None,
        progressive = None,
        chunk_duration : Optional[int] = None,
        log_distributed_frequencies= False,
        eps_stability: float = 1e-3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        re_lower: float = None,
        re_upper: float = None,
        ensure_stability: str = 'abs',
        sigmoid_scale: float = 1.0,
        structured_initialisation = False
    ):
        super(Trainable_decoder, self).__init__()

        self.magssm_decoder = MagSSM_Decoder(
            n_fft = n_fft,
            n_hop = n_hop,
            length = length,
            dim_state = dim_state,
            og = og,
            B_C_init = B_C_init,
            C_C_init = C_C_init,
            device = device,
            progressive = progressive,
            log_distributed_frequencies = log_distributed_frequencies,
            chunk_duration = chunk_duration,
            eps_stability = eps_stability,
            dt_min = dt_min,
            dt_max = dt_max,
            re_lower = re_lower,
            re_upper = re_upper,
            ensure_stability = ensure_stability,
            sigmoid_scale = sigmoid_scale,
            structured_initialisation = structured_initialisation 
        ).to(device)

        self.length = length
        self.device = device
        self.return_complex_signal = return_complex_signal

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor: # (B,C,F,T,2) |----> (B,C,length)

        assert(self.length is not None, "Length is not defined for the decoder. Please set it in the constructor.")
        assert(x.ndim == 5, "Input should have 5 dimensions (B,C,F,T,2)")

        x_complex = torch.complex(x[...,0], x[...,1])

        B, C, F, T = x_complex.data.shape # expected type complex torch tensor


        y = torch.zeros(B,C,self.length).to(self.device)

        for idx_channel in range(C):
            y[:,idx_channel,:] = self.magssm_decoder(x_complex[:,idx_channel,:,:])

        if self.return_complex_signal:
            return y

        return y.real
