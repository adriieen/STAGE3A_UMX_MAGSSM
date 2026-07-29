"""Benchmark script to quantify training convergence (5 epochs) for:
 1. Baseline (Standard Orthogonal B/C, No PC)
 2. Proposed Method (Dynamic Phase Correction on C + Window Scaling on B)

n_fft=30, n_hop=5, nb_states=32. Runs on CPU in ~10 seconds total.
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


def create_decoder(n_fft=30, n_hop=5, nb_states=32, window=None, phase_correction=False, seed=42):
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


def run_training_experiment(phase_correction=False, epochs=5, num_batches=8, seed=42):
    n_fft = 30
    n_hop = 5
    nb_states = 32
    sample_rate = 16000
    signal_len = 1200

    window = get_regularised_window(n_fft=n_fft, epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85)
    stft, _ = make_filterbanks(n_fft=n_fft, n_hop=n_hop, sample_rate=sample_rate, regularize=True, epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85)
    encoder = stft.to("cpu")
    encoder.eval()

    dec = create_decoder(n_fft, n_hop, nb_states, window=window, phase_correction=phase_correction, seed=seed)
    optimizer = torch.optim.AdamW(dec.parameters(), lr=0.005)

    # Generate fixed training dataset batches
    seed_all(seed + 100)
    batches = []
    for _ in range(num_batches):
        t = torch.linspace(0, signal_len / sample_rate, signal_len)
        f1, f2 = np.random.uniform(200, 800), np.random.uniform(800, 1500)
        x = torch.sin(2 * math.pi * f1 * t) + 0.5 * torch.sin(2 * math.pi * f2 * t) + 0.1 * torch.randn(signal_len)
        x = x.unsqueeze(0).unsqueeze(0)
        batches.append(x)

    epoch_losses = []
    # Epoch 0: Initial loss
    dec.eval()
    with torch.no_grad():
        init_loss = 0.0
        for x in batches:
            X = encoder(x)
            y_hat = dec(X, length=signal_len)
            Y_hat = encoder(y_hat)
            init_loss += F.mse_loss(Y_hat, X).item()
        init_loss /= len(batches)
    epoch_losses.append(init_loss)

    # Training loop for 5 epochs
    for ep in range(1, epochs + 1):
        dec.train()
        ep_loss = 0.0
        for x in batches:
            optimizer.zero_grad()
            with torch.no_grad():
                X = encoder(x)
            y_hat = dec(X, length=signal_len)
            Y_hat = encoder(y_hat)
            loss = F.mse_loss(Y_hat, X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(batches)
        epoch_losses.append(ep_loss)

    return epoch_losses


def main():
    print("\n" + "=" * 75)
    print("DYNAMIC PHASE CORRECTION CONVERGENCE BENCHMARK (5 Epochs)")
    print("=" * 75)

    hist_base = run_training_experiment(phase_correction=False, epochs=5, seed=42)
    hist_dyn  = run_training_experiment(phase_correction=True, epochs=5, seed=42)

    print("\n--- 1. Baseline (Standard Orthogonal B/C) ---")
    print(f"  Epoch 0 (Init)  : MSE = {hist_base[0]:.6f}")
    print(f"  Epoch 1         : MSE = {hist_base[1]:.6f}")
    print(f"  Epoch 3         : MSE = {hist_base[3]:.6f}")
    print(f"  Epoch 5         : MSE = {hist_base[5]:.6f}")

    print("\n--- 2. Proposed Method (Dynamic PC on C + Window B) ---")
    print(f"  Epoch 0 (Init)  : MSE = {hist_dyn[0]:.6f}")
    print(f"  Epoch 1         : MSE = {hist_dyn[1]:.6f}")
    print(f"  Epoch 3         : MSE = {hist_dyn[3]:.6f}")
    print(f"  Epoch 5         : MSE = {hist_dyn[5]:.6f}")

    print("\n" + "=" * 75)
    print("SUMMARY COMPARISON (Initial vs Epoch 5 Loss):")
    print("=" * 75)
    diff_init = (hist_dyn[0] - hist_base[0]) / hist_base[0] * 100
    diff_ep1  = (hist_dyn[1] - hist_base[1]) / hist_base[1] * 100
    diff_ep5  = (hist_dyn[5] - hist_base[5]) / hist_base[5] * 100

    print(f"Epoch 0 (Init) : Baseline = {hist_base[0]:.5f} | Proposed = {hist_dyn[0]:.5f} ({diff_init:+6.2f}%)")
    print(f"Epoch 1        : Baseline = {hist_base[1]:.5f} | Proposed = {hist_dyn[1]:.5f} ({diff_ep1:+6.2f}%)")
    print(f"Epoch 5        : Baseline = {hist_base[5]:.5f} | Proposed = {hist_dyn[5]:.5f} ({diff_ep5:+6.2f}%)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
