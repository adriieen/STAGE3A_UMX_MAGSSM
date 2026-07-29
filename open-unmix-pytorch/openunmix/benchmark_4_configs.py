"""Comparative Benchmark Script (5 Epochs) testing the 4 requested configurations + Baseline:
 0) Baseline: B non modifiée, Pas de correction de phase
 1) B non modifiée, Phase statique
 2) B non modifiée, Phase dynamique
 3) Fenêtrage B, Phase statique
 4) Fenêtrage B, Phase dynamique

n_fft=30, n_hop=5, nb_states=32. Fast execution (~15 seconds).
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


def run_experiment(name, window_for_b=None, phase_correction_mode=False, epochs=5, num_batches=8, seed=42):
    n_fft = 30
    n_hop = 5
    nb_states = 32
    sample_rate = 16000
    signal_len = 1200

    # Decoder window passed for COLA normalization in MagSSM_Decoder
    cola_window = get_regularised_window(n_fft=n_fft, epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85)
    stft, _ = make_filterbanks(n_fft=n_fft, n_hop=n_hop, sample_rate=sample_rate, regularize=True, epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85)
    encoder = stft.to("cpu")
    encoder.eval()

    dec = create_decoder(n_fft, n_hop, nb_states, window=cola_window, phase_correction=phase_correction_mode, seed=seed)

    # If window_for_b is None, remove B window scaling from SSM
    if window_for_b is None:
        mimo = dec.magssm_decoder.single_sequence_decoder.mimo.seq
        # Re-initialize B orthogonally without window scaling
        seed_all(seed)
        gain = np.sqrt(4/12)
        B_r = torch.empty(nb_states, mimo.B.shape[1])
        B_i = torch.empty(nb_states, mimo.B.shape[1])
        B_r = torch.nn.init.orthogonal_(B_r.T, gain=gain).T
        B_i = torch.nn.init.orthogonal_(B_i.T, gain=gain).T
        mimo.B.data = torch.stack((B_r, B_i), dim=-1)

    optimizer = torch.optim.AdamW(dec.parameters(), lr=0.005)

    # Fixed dataset batches
    seed_all(seed + 100)
    batches = []
    for _ in range(num_batches):
        t = torch.linspace(0, signal_len / sample_rate, signal_len)
        f1, f2 = np.random.uniform(200, 800), np.random.uniform(800, 1500)
        x = torch.sin(2 * math.pi * f1 * t) + 0.5 * torch.sin(2 * math.pi * f2 * t) + 0.1 * torch.randn(signal_len)
        x = x.unsqueeze(0).unsqueeze(0)
        batches.append(x)

    epoch_losses = []

    # Epoch 0 (Init loss)
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

    # 5 training epochs
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
    cola_window = get_regularised_window(n_fft=30, epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85)

    configs = [
        ("0) Baseline (B non modifiée, Pas de correction phase)", None, False),
        ("1) Matrice B non modifiée, Phase statique", None, "static"),
        ("2) Matrice B non modifiée, Phase dynamique", None, "dynamic"),
        ("3) Fenêtrage B et Phase statique", cola_window, "static"),
        ("4) Fenêtrage B et Phase dynamique", cola_window, "dynamic"),
    ]

    print("\n" + "=" * 80)
    print("COMPARATIVE 5-EPOCH BENCHMARK (n_fft=30, n_hop=5, nb_states=32)")
    print("=" * 80)

    results = {}
    for name, win, pc_mode in configs:
        losses = run_experiment(name, window_for_b=win, phase_correction_mode=pc_mode, epochs=5, seed=42)
        results[name] = losses

    print(f"{'Configuration':50s} | {'Init (Ep0)':10s} | {'Epoch 1':10s} | {'Epoch 3':10s} | {'Epoch 5':10s}")
    print("-" * 96)
    for name in results:
        l = results[name]
        print(f"{name:50s} | {l[0]:10.5f} | {l[1]:10.5f} | {l[3]:10.5f} | {l[5]:10.5f}")

    print("=" * 96)

    base_ep5 = results["0) Baseline (B non modifiée, Pas de correction phase)"][5]
    base_init = results["0) Baseline (B non modifiée, Pas de correction phase)"][0]
    print("\nRelative Comparison vs Baseline:")
    for name in results:
        diff_init = (results[name][0] - base_init) / base_init * 100
        diff_ep5 = (results[name][5] - base_ep5) / base_ep5 * 100
        print(f" -> {name:48s} | Init: {diff_init:+6.2f}% | Ep5: {diff_ep5:+6.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
