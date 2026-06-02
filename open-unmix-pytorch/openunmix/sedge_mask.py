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
from magssm import MagSSM
from utils_edge_var import LogNormalizer

sys.path.append('/home/adubois/openunmix/OpenUnmix/SEdge/src')
from model_edge.sequence_musdbadrien import sedge_sequence

class SedgeMask(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
        encoder (nn.Module, optional): The STFT encoder. If None, a default STFT encoder will be used.
        hidden_size_factors (list or None): Factors by which to reduce the hidden size of each layer. 
            If None, the hidden size of each layer will be the same as hidden_size (Default: `None`).
        output_size_factors (list or None): Factors by which to reduce the output size of each layer. 
            If None, the output size of each layer will be the same as hidden_size (Default: `None`).
        nb_layers (int): Number of layers in the main separation module (Default: `3`).
    """

    def __init__(
        self,
        nb_bins: int = 2049,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        input_mean: Optional[np.ndarray] = None,
        input_scale: Optional[np.ndarray] = None,
        hidden_size_factors = None,
        output_size_factors = None,
        n_fft = 4096,
        n_hop = 1024,
        dim_state = 129,
        d_out = 129,
        encoder : Optional[nn.Module] = None,
        device = None,
        use_edge = False,
        unidirectional = True,
        use_magssm = False,
        chunk_duration : Optional[int] = None,
        mel = False
        
    ):
        super(SedgeMask, self).__init__()

        self.nb_output_bins = nb_bins

        self.nb_channels = nb_channels
        self.hidden_size = hidden_size
        self.use_edge = use_edge

        self.fc0 = Linear(nb_bins, d_out)

        #self.fc1 = Linear(self.nb_bins * nb_channels, hidden_size, bias=False)
        self.fc1 = Linear(d_out * nb_channels, hidden_size, bias=False)
        self.bn1 = BatchNorm1d(hidden_size)

        if hidden_size_factors == None : 
            hidden_size_factors = np.array([1 for _ in range(int(nb_layers))]) 
        else : hidden_size_factors = np.array(hidden_size_factors)

        if output_size_factors == None :
            output_size_factors = np.array([1 for _ in range(int(nb_layers))])
        else : output_size_factors = np.array(output_size_factors)

        self.sedge = sedge_sequence(
            input_size=hidden_size,
            hidden_sizes = list((hidden_size // hidden_size_factors).astype(int)),
            output_sizes = list((hidden_size // output_size_factors).astype(int)),
            dropout=0.4
        )


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
        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins * nb_channels,
            bias=False,
        )
        self.bn3 = BatchNorm1d(self.nb_output_bins * nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(-input_mean).float()
        else:
            input_mean = torch.zeros(nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(1.0 / input_scale).float()
        else:
            input_scale = torch.ones(nb_bins)


        self.LogNormalizer = LogNormalizer(nb_bins, d_out, 
                                           linear_neg_mean= input_mean,
                                           linear_inv_std= input_scale)


        self.output_scale = Parameter(torch.ones(self.nb_output_bins).float())
        self.output_mean = Parameter(torch.ones(self.nb_output_bins).float())

        self.encoder = encoder
        self.device = device

        self.magssm = MagSSM(
            dim_state = dim_state,
            d_out = d_out,
            device = device,
            mel = mel,
            chunk_duration = chunk_duration,
            subsampling_factor = n_hop,

        ).to(device)



    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        if self.encoder : X = self.encoder(x)
        else : raise ValueError('Encoder should not be none')


        _ , _, _, T = X.data.shape
        x = self.magssm(x)
        x = torch.abs(x)    # B, 2, (d_out = nb_magssm_states), nb_samples

        # print("Sortie de magssm = ", x.data.shape) 
        
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape # nb_bins = d_out

        x = self.LogNormalizer(x)


        x = self.fc1(x.reshape(-1, nb_channels * nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)

        # print("entrée bottleneck = ", x.data.shape)
        
        if self.use_edge : 
            sequence_out = self.sedge(x)
        
        else :
            sequence_out = self.lstm(x) 



        x = torch.cat([x, sequence_out], -1)

        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        # print("input FC3 = ", x.data.shape)


        x = self.fc3(x)
        x = self.bn3(x)

        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x *= self.output_scale
        x += self.output_mean

        x = x.permute(1, 2, 3, 0)
        x = x[..., :T]

        # print("dim du mask = ", x.data.shape)


        return (F.relu(x) * X)


class Separator(nn.Module):
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
        device = None
    ):
        super(Separator, self).__init__()

        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft, self.istft = make_filterbanks(
            n_fft=n_fft,
            n_hop=n_hop,
            method=filterbank,
            sample_rate=sample_rate,
        )
        self.complexnorm = ComplexNorm(mono=nb_channels == 1)

        self.target_models = nn.ModuleDict(target_models)
        self.nb_targets = len(self.target_models)
        self.register_buffer("sample_rate", torch.as_tensor(sample_rate))
        self.device = device

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
            
            actual_device = next(target_module.parameters()).device
            print("modele sur ", actual_device)
            
            audio_device = audio.detach().clone().to(self.device)
            target_spectrogram = target_module(audio_device)
            spectrograms[..., j] = target_spectrogram

        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        if self.residual:
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception(
                "Cannot use EM if only one target is estimated."
                "Provide two targets or create an additional "
                "one with `--residual`"
            )

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