from typing import Optional

import torch
import torchaudio
from torch import Tensor
import torch.nn as nn

try:
    from asteroid_filterbanks.enc_dec import Encoder, Decoder
    from asteroid_filterbanks.transforms import to_torchaudio, from_torchaudio
    from asteroid_filterbanks import torch_stft_fb
except ImportError:
    pass


def get_regularised_window(
    n_fft: int,
    epsilon1: float = 0.07,
    lambda_coeff_1: float = 0.77,
    lambda_coeff_2: float = 0.85,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a Hann window regularised with a bilateral double-exponential envelope.

    The returned tensor is::

        w_reg = hann(n_fft) * (eps1 * exp(-l1 * |t - N/2|)
                              + eps2 * exp(-l2 * |t - N/2|))

    where ``eps2 = 1 - eps1``, ``l1 = lambda_coeff_1 / n_fft``,
    ``l2 = lambda_coeff_2 / n_fft``, and ``|t - N/2|`` is the distance
    to the window centre.

    The two exponential terms provide independent control over the
    fast-decay (sidelobe suppression) and slow-decay (main-lobe shape)
    components while preserving the symmetric OLA property of the Hann
    window.

    Args:
        n_fft:          Window (and FFT) size.
        epsilon1:       Weight of the fast-decay exponential (eps2 = 1 - eps1).
        lambda_coeff_1: Coefficient for the fast decay (lambda_val_1 = lambda_coeff_1 / n_fft).
        lambda_coeff_2: Coefficient for the slow decay (lambda_val_2 = lambda_coeff_2 / n_fft).
        device:         Torch device for the output tensor.
        dtype:          Floating-point dtype (default ``torch.float32``).

    Returns:
        Tensor of shape ``(n_fft,)`` with ``requires_grad=False``.
    """
    hann = torch.hann_window(n_fft, device=device, dtype=dtype)
    t = torch.arange(n_fft, device=device, dtype=dtype)
    t_centered = torch.abs(t - n_fft // 2)

    epsilon2 = 1.0 - epsilon1
    lambda_val_1 = lambda_coeff_1 / n_fft
    lambda_val_2 = lambda_coeff_2 / n_fft

    w_exp_rapide = epsilon1 * torch.exp(-lambda_val_1 * t_centered)
    w_exp_lente = epsilon2 * torch.exp(-lambda_val_2 * t_centered)

    window = hann * (w_exp_rapide + w_exp_lente)
    window = window.detach().requires_grad_(False)
    return window


def make_filterbanks(
    n_fft=4096,
    n_hop=1024,
    center=False,
    sample_rate=44100.0,
    method="torch",
    regularize=False,
    epsilon1=0.07,
    lambda_coeff_1=0.77,
    lambda_coeff_2=0.85,
):
    if regularize:
        window = get_regularised_window(
            n_fft, epsilon1=epsilon1,
            lambda_coeff_1=lambda_coeff_1, lambda_coeff_2=lambda_coeff_2,
        )
    else:
        window = torch.hann_window(n_fft) + 1e-4

    if method == "torch":
        encoder = TorchSTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
        decoder = TorchISTFT(n_fft=n_fft, n_hop=n_hop, window=window, center=center)
    elif method == "asteroid":
        fb = torch_stft_fb.TorchSTFTFB.from_torch_args(
            n_fft=n_fft,
            hop_length=n_hop,
            win_length=n_fft,
            window=window,
            center=center,
            sample_rate=sample_rate,
        )
        encoder = AsteroidSTFT(fb)
        decoder = AsteroidISTFT(fb)
    else:
        raise NotImplementedError
    return encoder, decoder


class AsteroidSTFT(nn.Module):
    def __init__(self, fb):
        super(AsteroidSTFT, self).__init__()
        self.enc = Encoder(fb)

    def forward(self, x):
        aux = self.enc(x)
        return to_torchaudio(aux)


class AsteroidISTFT(nn.Module):
    def __init__(self, fb):
        super(AsteroidISTFT, self).__init__()
        self.dec = Decoder(fb)

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        aux = from_torchaudio(X)
        return self.dec(aux, length=length)


class TorchSTFT(nn.Module):
    """Multichannel Short-Time-Fourier Forward transform
    uses hard coded hann_window.
    Args:
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        window (nn.Parameter, optional): window function
    """

    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        window: Optional[nn.Parameter] = None,
    ):
        super(TorchSTFT, self).__init__()
        if window is None:
            self.register_buffer('window', torch.hann_window(n_fft))
        else:
            # Detach to avoid DDP treating it as a trainable parameter
            self.register_buffer('window', window.detach() if isinstance(window, torch.Tensor) else window)

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x: Tensor) -> Tensor:
        """STFT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            STFT (Tensor): complex stft of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """

        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape
        # print("input encoder", shape)

        # pack batch
        x = x.view(-1, shape[-1])

        complex_stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        stft_f = torch.view_as_real(complex_stft)
        # unpack batch
        stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])

        # print("output encoder", stft_f.data.shape)

        return stft_f


class TorchISTFT(nn.Module):
    """Multichannel Inverse-Short-Time-Fourier functional
    wrapper for torch.istft to support batches
    Args:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        window (callable, optional): window function
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        length (int, optional): audio signal length to crop the signal
    Returns:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    """

    def __init__(
        self,
        n_fft: int = 4096,
        n_hop: int = 1024,
        center: bool = False,
        sample_rate: float = 44100.0,
        window: Optional[nn.Parameter] = None,
    ) -> None:
        super(TorchISTFT, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate

        if window is None:
            self.register_buffer('window', torch.hann_window(n_fft) + 1e-4)
        else:
            self.register_buffer('window', window.detach() if isinstance(window, torch.Tensor) else window)

    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])

        y = torch.istft(
            torch.view_as_complex(X),
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )

        y = y.reshape(shape[:-3] + y.shape[-1:])

        return y


class ComplexNorm(nn.Module):
    r"""Compute the norm of complex tensor input.

    Extension of `torchaudio.functional.complex_norm` with mono

    Args:
        mono (bool): Downmix to single channel after applying power norm
            to maximize
    """

    def __init__(self, mono: bool = False):
        super(ComplexNorm, self).__init__()
        self.mono = mono

    def forward(self, spec: Tensor) -> Tensor:
        """
        Args:
            spec: complex_tensor (Tensor): Tensor shape of
                `(..., complex=2)`

        Returns:
            Tensor: Power/Mag of input
                `(...,)`
        """
        # take the magnitude

        spec = torch.abs(torch.view_as_complex(spec))

        # downmix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)

        return spec


