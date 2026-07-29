from typing import Optional, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from transforms import make_filterbanks, ComplexNorm
from spectrogram import Trainable_spectrogram
from decoder import Trainable_decoder

class MagSSM_OpenUnmix(nn.Module):
    """OpenUnmix core separation module using trainable MagSSM encoder.

    Args:
        nb_bins (int): Number of frequency bins (Default: `2049`).
        nb_channels (int): Number of audio channels (Default: `2`).
        hidden_size (int): Bottleneck layer hidden size (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model (Default: `False`).
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
        input_scale (ndarray or None): global data scale of shape `(nb_bins, )`.
        max_bin (int or None): internal frequency bin threshold.
        dim_state (int): Number of hidden states in MagSSM.
        d_out (int or None): output dimension of MagSSM encoder.
        n_fft (int): FFT size.
        n_hop (int): Hop size.
        encoder (nn.Module, optional): STFT encoder module.
        device: Device (cpu/cuda).
        chunk_duration (int or None): chunk duration in samples for Progressive SSM.
        log_distributed_frequencies (bool): initialize SSM frequencies logarithmically.
        use_layernorm (bool): Use LayerNorm instead of BatchNorm1d.
    """

    def __init__(
        self,
        nb_bins: int = 2049,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        unidirectional: bool = False,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        max_bin: Optional[int] = None,
        dim_state: int = 129,
        d_out: Optional[int] = None,
        n_fft: int = 4096,
        n_hop: int = 1024,
        encoder: Optional[nn.Module] = None,
        device = None,
        chunk_duration: Optional[int] = None,
        log_distributed_frequencies: bool = False,
        use_layernorm: bool = True,
        og: bool = False,
        eps_stability: float = 1e-3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        re_lower: Optional[float] = None,
        re_upper: Optional[float] = None,
        ensure_stability: str = 'abs',
        sigmoid_scale: float = 1.0,
        progressive: bool = False,
        fft_kernel: bool = False,
        complex_spectrogram: bool = True,
        structured_initialisation: bool = False,
    ):
        super(MagSSM_OpenUnmix, self).__init__()

        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size
        self.nb_channels = nb_channels
        self.use_layernorm = use_layernorm
        
        # d_out is the output frequency dimension from the SSM
        self.d_out = d_out if d_out is not None else self.nb_bins

        # spectrogram
        self.trainable_spectrogram = Trainable_spectrogram(
            nb_bins=n_fft // 2 + 1,
            nb_channels=nb_channels,
            n_hop=n_hop,
            dim_state=dim_state,
            og=og,
            B_C_init="orthogonal" if og else "ones",
            encoder=encoder,
            device=device,
            progressive=progressive,
            fft_kernel=fft_kernel,
            chunk_duration=chunk_duration,
            log_distributed_frequencies=log_distributed_frequencies,
            eps_stability=eps_stability,
            dt_min=dt_min,
            dt_max=dt_max,
            re_lower=re_lower,
            re_upper=re_upper,
            ensure_stability=ensure_stability,
            sigmoid_scale=sigmoid_scale,
            complex_spectrogram=complex_spectrogram,
            structured_initialisation=structured_initialisation,
        ).to(device)

        self.device = device

        # Separation module projection
        self.fc1 = Linear(self.d_out * nb_channels, hidden_size, bias=False)

        if use_layernorm:
            self.ln1 = nn.LayerNorm(hidden_size)
        else:
            self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(in_features=fc2_hiddensize, out_features=hidden_size, bias=False)

        if use_layernorm:
            self.ln2 = nn.LayerNorm(hidden_size)
        else:
            self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )

        if use_layernorm:
            self.ln3 = nn.LayerNorm(self.nb_output_bins * nb_channels)
        else:
            self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        # Setup input scaling parameters
        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean[: self.nb_bins]).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale[: self.nb_bins]).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

    def freeze(self):
        # helper to freeze everything
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: raw audio waveform of shape `(nb_samples, nb_channels, nb_timesteps)`
        
        Returns:
            complex spectrogram of the target estimate of shape `(nb_samples, nb_channels, nb_bins, nb_frames, 2)` or `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        # C-MAGSSM complex spectrogram output
        spec = self.trainable_spectrogram(x)

        if spec.ndim == 5:
            spec_complex = torch.complex(spec[..., 0], spec[..., 1])
        else:
            spec_complex = spec

        # Magnitude of mixture spectrogram
        magnitude = torch.abs(spec_complex)

        # Permute to T, B, C, F for OpenUnmix processing
        x_in = magnitude.permute(3, 0, 1, 2)

        nb_frames, nb_samples, nb_channels, d_out = x_in.data.shape

        # Crop frequency dimension if needed
        x_in = x_in[..., : self.nb_bins]

        # Shift and scale input
        x_in = x_in + self.input_mean
        x_in = x_in * self.input_scale

        # Project and normalize (Encoder block: FC1 + LN1/BN1 + Tanh)
        h = self.fc1(x_in.reshape(-1, nb_channels * self.nb_bins))
        if self.use_layernorm:
            h = self.ln1(h)
        else:
            h = self.bn1(h)
        
        h = h.reshape(nb_frames, nb_samples, self.hidden_size)
        h = torch.tanh(h)

        # Run Bidirectional/Unidirectional LSTM (Separator block)
        if not h.is_contiguous():
            h = h.contiguous()

        if h.shape[0] >= 65536:
            prev_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            try:
                lstm_out, _ = self.lstm(h)
            finally:
                torch.backends.cudnn.enabled = prev_enabled
        else:
            lstm_out, _ = self.lstm(h)

        # Skip connection
        h_cat = torch.cat([h, lstm_out], -1)

        # FC2 layer + LN2/BN2 + ReLU
        h2 = self.fc2(h_cat.reshape(-1, h_cat.shape[-1]))
        if self.use_layernorm:
            h2 = self.ln2(h2)
        else:
            h2 = self.bn2(h2)
        h2 = F.relu(h2)

        # FC3 layer + LN3/BN3 (Decoder block)
        out = self.fc3(h2)
        if self.use_layernorm:
            out = self.ln3(out)
        else:
            out = self.bn3(out)

        # Reshape to spectrogram mask
        mask = out.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        mask *= self.output_scale
        mask += self.output_mean
        mask = F.relu(mask)

        # Permute back to B, C, F, T
        mask = mask.permute(1, 2, 3, 0)

        # Pointwise product: Target estimated complex spectrogram = Mask * Mix Spectrogram
        if spec.ndim == 5:
            target_spec = mask[..., None] * spec
        else:
            target_spec = mask * spec_complex

        return target_spec


class Separator(nn.Module):
    """Wrapper to enable end-to-end separation using MagSSM_OpenUnmix and Trainable_decoder."""
    def __init__(
        self,
        target_models: Mapping[str, nn.Module],
        decoders: Optional[Mapping[str, nn.Module]] = None,
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: float = 44100.0,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300,
        filterbank: str = "torch",
        device = None,
        regularize: bool = False,
        epsilon1: float = 0,
        lambda_coeff_1: float = 0,
        lambda_coeff_2: float = 0,
    ):
        super(Separator, self).__init__()

        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len
        self.device = device

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            method=filterbank,
            sample_rate=sample_rate,
            regularize=regularize,
            epsilon1=epsilon1,
            lambda_coeff_1=lambda_coeff_1,
            lambda_coeff_2=lambda_coeff_2,
        )

        self.complexnorm = ComplexNorm(mono=nb_channels == 1)
        self.target_models = nn.ModuleDict(target_models)
        if decoders is not None:
            self.decoders = nn.ModuleDict(decoders)
        else:
            self.decoders = None

        self.nb_targets = len(self.target_models)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        """
        Performing the separation on audio input.

        Args:
            audio: mixture audio waveform (nb_samples, nb_channels, nb_timesteps)

        Returns:
            stacked tensor of separated waveforms (nb_samples, nb_targets, nb_channels, nb_timesteps)
        """
        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        estimates = []
        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            target_module.to(self.device)
            audio_device = audio.to(self.device)

            # 1. Forward through MagSSM_OpenUnmix model -> target complex spectrogram
            S_target = target_module(audio_device)

            # 2. Forward through Trainable_decoder if present
            if self.decoders is not None and target_name in self.decoders:
                decoder = self.decoders[target_name].to(self.device)
                y_hat = decoder(S_target, length=audio.shape[-1])
            else:
                # Fallback to inverse STFT if no trainable decoder provided
                if S_target.ndim == 5:
                    y_hat = self.istft(S_target.permute(0, 1, 2, 3, 4), length=audio.shape[-1])
                else:
                    y_hat = self.istft(S_target, length=audio.shape[-1])

            estimates.append(y_hat)

        estimates = torch.stack(estimates, dim=1) # (nb_samples, nb_targets, nb_channels, nb_timesteps)

        # Pad or crop to match original length
        pad_len = audio.shape[2] - estimates.shape[-1]
        if pad_len > 0:
            estimates = torch.nn.functional.pad(estimates, (0, pad_len))
        elif pad_len < 0:
            estimates = estimates[..., :audio.shape[2]]

        return estimates

    def to_dict(self, estimates: Tensor, aggregate_dict: Optional[dict] = None) -> dict:
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        if self.residual:
            estimates_dict["residual"] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = torch.tensor(0.0)
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict

