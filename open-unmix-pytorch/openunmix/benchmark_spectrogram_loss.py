"""Benchmark script to evaluate Spectrogram Reconstruction Loss MSE(Encoder(Decoder(X)), X)
with DYNAMIC Phase Correction on C and Window Scaling on B.

Runs on CPU in ~5 seconds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from path_config import setup_paths
setup_paths()

import torch
import torch.nn.functional as F
import numpy as np
import math

from decoder import Trainable_decoder
from transforms import get_regularised_window, make_filterbanks


def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_decoder(
    n_fft=30,
    n_hop=5,
    nb_states=32,
    window=None,
    phase_correction=False,
    seed=42
):
    seed_all(seed)
    decoder = Trainable_decoder(
        n_fft=n_fft,
        n_hop=n_hop,
        window=window,
        dim_state=nb_states,
        og=False,
        B_C_init="orthogonal",
        device="cpu",
        chunk_duration=100,
        log_distributed_frequencies=False,
        eps_stability=0,
        dt_min=0.005,
        dt_max=0.5,
        ensure_stability="abs",
        structured_initialisation=True,
        phase_correction=phase_correction
    )
    return decoder


def benchmark():
    n_fft = 30
    n_hop = 5
    nb_states = 32
    sample_rate = 16000
    signal_len = 1600

    window = get_regularised_window(n_fft=n_fft, epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85)
    stft, _ = make_filterbanks(n_fft=n_fft, n_hop=n_hop, sample_rate=sample_rate, regularize=True, epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85)
    encoder = stft.to("cpu")
    encoder.eval()

    seed_all(100)
    t = torch.linspace(0, signal_len / sample_rate, signal_len)
    sig1 = torch.sin(2 * math.pi * 440 * t) + 0.5 * torch.sin(2 * math.pi * 880 * t) + 0.25 * torch.sin(2 * math.pi * 1320 * t)
    sig2 = torch.zeros(signal_len); sig2[::200] = 1.0
    sig3 = torch.randn(signal_len)
    sig4 = torch.sin(2 * math.pi * (100 + 1000 * t) * t)
    sig5 = sig1 + 0.5 * sig3

    batch_x = torch.stack([sig1, sig2, sig3, sig4, sig5], dim=0).unsqueeze(1)

    with torch.no_grad():
        X = encoder(batch_x)

    # 1. Baseline
    dec_base = create_decoder(n_fft, n_hop, nb_states, window=window, phase_correction=False, seed=42)
    with torch.no_grad():
        y_base = dec_base(X, length=signal_len)
        Y_base = encoder(y_base)
        loss_base = F.mse_loss(Y_base, X).item()

    # 2. Dynamic Phase Correction + Window Scaling
    dec_full = create_decoder(n_fft, n_hop, nb_states, window=window, phase_correction=True, seed=42)
    with torch.no_grad():
        y_full = dec_full(X, length=signal_len)
        Y_full = encoder(y_full)
        loss_full = F.mse_loss(Y_full, X).item()

    print("\n" + "=" * 75)
    print(f"BENCHMARK SPECTROGRAM RECONSTRUCTION LOSS (DYNAMIC PHASE CORRECTION)")
    print(f"Parameters: n_fft={n_fft}, n_hop={n_hop}, nb_states={nb_states}, signal_len={signal_len}")
    print("=" * 75)
    print(f" 1. Baseline (Standard Orthogonal B/C)      : MSE = {loss_base:.6f}")
    print(f" 2. Proposed Method (Dynamic PC + Window B) : MSE = {loss_full:.6f}")
    print("=" * 75)

    red_full = (loss_base - loss_full) / loss_base * 100
    print(f"\nRelative MSE Loss Reduction at Initialization: {red_full:+.2f}%")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    benchmark()
