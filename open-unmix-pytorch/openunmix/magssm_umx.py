from typing import Optional, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from transforms import make_filterbanks, ComplexNorm
from magssm import MagSSM_Encoder

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
        dim_state: int = 682,
        d_out: Optional[int] = None,
        n_fft: int = 682,
        n_hop: int = 34,
        encoder: Optional[nn.Module] = None,
        device = None,
        chunk_duration: Optional[int] = None,
        log_distributed_frequencies: bool = False,
        use_layernorm: bool = False,
        og: bool = False,
        eps_stability: float = 1e-3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        re_lower: float = None,
        re_upper: float = None,
        ensure_stability: str = 'abs',
        sigmoid_scale: float = 1.0,
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

        # MagSSM Encoder
        self.magssm_encoder = MagSSM_Encoder(
            d_in=1,
            dim_state=dim_state,
            d_out=self.d_out,
            og=og,
            device=device,
            log_distributed_frequencies=log_distributed_frequencies,
            chunk_duration=chunk_duration,
            subsampling_factor=n_hop,
            eps_stability=eps_stability,
            dt_min=dt_min,
            dt_max=dt_max,
            re_lower=re_lower,
            re_upper=re_upper,
            ensure_stability=ensure_stability,
            sigmoid_scale=sigmoid_scale,
        ).to(device)

        # STFT encoder for reference/masking
        self.encoder = encoder
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

    def forward(self, x: Tensor, X: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: raw audio waveform of shape `(nb_samples, nb_channels, nb_timesteps)`
            X: Optional reference spectrogram of shape `(nb_samples, nb_channels, nb_bins, nb_frames)`
        
        Returns:
            masked target spectrogram of shape `(nb_samples, nb_channels, nb_bins, nb_frames)`
        """
        x_orig = x
        if X is None:
            with torch.no_grad():
                if self.encoder:
                    X = self.encoder(x_orig.float())
                else:
                    raise ValueError('Encoder should not be none')

        _, _, _, T = X.data.shape

        # Run MagSSM Encoder channel-wise
        x_left, x_right = x[:, 0, :], x[:, 1, :]
        x_left = self.magssm_encoder(x_left)
        x_right = self.magssm_encoder(x_right)

        # Combine channels and absolute value
        x = torch.cat((x_left[:, None, ...], x_right[:, None, ...]), dim=1) # B, 2, T, d_out
        x = torch.abs(x)

        # Permute to T, B, C, F
        x = x.permute(2, 0, 1, 3)

        nb_frames, nb_samples, nb_channels, d_out = x.data.shape

        # Crop frequency dimension if needed
        x = x[..., : self.nb_bins]

        # Shift and scale input
        x = x + self.input_mean
        x = x * self.input_scale

        # Project and normalize
        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        if self.use_layernorm:
            x = self.ln1(x)
        else:
            x = self.bn1(x)
        
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)

        # Run Bidirectional/Unidirectional LSTM
        if not x.is_contiguous():
            x = x.contiguous()

        if x.shape[0] >= 65536:
            prev_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            try:
                lstm_out, _ = self.lstm(x)
            finally:
                torch.backends.cudnn.enabled = prev_enabled
        else:
            lstm_out, _ = self.lstm(x)

        # Skip connection
        x = torch.cat([x, lstm_out], -1)

        # FC2 layer
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        if self.use_layernorm:
            x = self.ln2(x)
        else:
            x = self.bn2(x)
        x = F.relu(x)

        # FC3 layer
        x = self.fc3(x)
        if self.use_layernorm:
            x = self.ln3(x)
        else:
            x = self.bn3(x)

        # Reshape to spectrogram
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x *= self.output_scale
        x += self.output_mean

        # Permute back to B, C, F, T
        x = x.permute(1, 2, 3, 0)
        x = x[..., :T]

        # Return masked spectrogram
        return F.relu(x) * X


class Separator(nn.Module):
    """Wrapper to enable separation of target sources."""
    def __init__(
        self,
        target_models: Mapping[str, nn.Module],
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
        epsilon1: float = 0.07,
        lambda_coeff_1: float = 0.77,
        lambda_coeff_2: float = 0.85,
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
        self.nb_targets = len(self.target_models)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio: Tensor) -> Tensor:
        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft).to(self.device)

        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

        for j, (target_name, target_module) in enumerate(self.target_models.items()):
            target_module.to(self.device)
            audio_device = audio.detach().clone().to(self.device)
            target_spectrogram = target_module(audio_device)
            spectrograms[..., j] = target_spectrogram

        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        if self.residual:
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception("Cannot use EM if only one target is estimated.")

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device)
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                from filtering import wiener
                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual,
                )

        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()
        estimates = self.istft(targets_stft, length=None)

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
