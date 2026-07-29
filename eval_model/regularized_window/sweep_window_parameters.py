#!/usr/bin/env python3
"""
Balayage systématique des paramètres de régularisation de fenêtre
sur toutes les combinaisons (hop_factor, alpha_ola).

Pour chaque config :
  1. Recherche des paramètres optimaux (epsilon1, lambda_1, lambda_2)
     via la grille de find_optimal_param.py (version contrainte log)
  2. Évaluation de l'erreur relative de modulation OLA (dB)
     via la logique de test_régul_window.py

Les résultats sont stockés dans un ResultsStore et sauvegardés en JSON.

Usage:
    python sweep_window_params.py [--output results.json]
"""
import argparse
import itertools
import time

import numpy as np

from results_store import ResultsStore


# ======================================================================
# 1. Recherche de paramètres optimaux  (logique de find_optimal_param.py)
# ======================================================================

def find_optimal_params(
    N: int,
    hop_factor: int,
    alpha_ola: float,
    beta_grad: float = 1.0,
):
    """
    Recherche sur grille les paramètres (epsilon1, lambda_coeff_1, lambda_coeff_2)
    minimisant la loss de régularité spectrale sous contrainte de validation OLA.
    """
    t = np.arange(N)
    n_hop = N // hop_factor
    w_hann = np.hanning(N)
    t_centered = np.abs(t - N // 2)

    # Grille d'exploration géométrique (logarithmique) pour cibler les zones critiques
    epsilon1_vals = np.logspace(np.log10(0.05), np.log10(0.5), 50)
    lambda_1_coeffs = np.logspace(np.log10(4), np.log10(15), 50)
    lambda_2_coeffs = np.logspace(np.log10(0.01), np.log10(4), 40)

    best_loss = float('inf')
    best_params = None

    # Variables de secours si aucun candidat ne descend sous le seuil strict de -80 dB
    fallback_loss = float('inf')
    fallback_params = None

    # Pré-calculs pour la réponse fréquentielle
    N_fft = 8192
    freqs = np.fft.fftfreq(N_fft)
    mask = freqs >= 0

    # Buffer OLA pré-alloué
    total_length = 4 * N
    hop_starts = list(range(0, total_length - N + 1, n_hop))

    for eps1, l1_c, l2_c in itertools.product(epsilon1_vals, lambda_1_coeffs, lambda_2_coeffs):
        eps2 = 1.0 - eps1
        l1 = l1_c / N
        l2 = l2_c / N

        # Construction de la fenêtre
        w_exp_rapide = eps1 * np.exp(-l1 * t_centered)
        w_exp_lente = eps2 * np.exp(-l2 * t_centered)
        w_reg = w_hann * (w_exp_rapide + w_exp_lente)

        # 1. Pénalité Overlap-Add (Peak-to-Peak et Erreur Relative)
        w_reg_sq = w_reg ** 2
        ola_buffer = np.zeros(total_length)
        for hs in hop_starts:
            ola_buffer[hs:hs + N] += w_reg_sq

        stable_zone = ola_buffer[N:2 * N]
        mean_ola = np.mean(stable_zone)
        ola_ptp = np.max(stable_zone) - np.min(stable_zone)
        relative_error_db = 20 * np.log10((ola_ptp / (mean_ola + 1e-12)) + 1e-12)

        # 2. Pénalité sur la réponse fréquentielle (norme du gradient + oscillations)
        W_reg = np.fft.fft(w_reg, N_fft)
        W_reg_mag = 20 * np.log10(np.abs(W_reg) + 1e-12)
        W_reg_mag_pos = W_reg_mag[mask]

        gradient = np.gradient(W_reg_mag_pos)
        grad_norm = np.linalg.norm(gradient)
        oscillation_penalty = np.sum(np.maximum(0, gradient))

        # Calcul des composantes de la fonction objective
        spectral_smoothness = beta_grad * (grad_norm + 2 * oscillation_penalty)
        combined_loss = alpha_ola * ola_ptp + spectral_smoothness

        # Strategie Convexe Contrainte : On cherche le plus lisse sous la barre des -80 dB
        if relative_error_db <= -80.0:
            if spectral_smoothness < best_loss:
                best_loss = spectral_smoothness
                best_params = (eps1, l1_c, l2_c)
        
        # Sauvegarde de la meilleure loss scalaire standard si la contrainte stricte est vide
        if combined_loss < fallback_loss:
            fallback_loss = combined_loss
            fallback_params = (eps1, l1_c, l2_c)

    if best_params is not None:
        return best_params[0], best_params[1], best_params[2], best_loss
    else:
        return fallback_params[0], fallback_params[1], fallback_params[2], fallback_loss


# ======================================================================
# 2. Évaluation de l'erreur relative de modulation  (logique de test_régul_window.py)
# ======================================================================

def compute_ola_relative_error(
    N: int,
    hop_factor: int,
    epsilon1: float,
    lambda_coeff_1: float,
    lambda_coeff_2: float,
) -> float:
    """
    Construit la fenêtre régularisée avec les paramètres donnés,
    simule l'OLA (somme des fenêtres au carré), et retourne l'erreur
    relative de modulation en dB.
    """
    t = np.arange(N)
    n_hop = N // hop_factor
    w_hann = np.hanning(N)
    t_centered = np.abs(t - N // 2)

    epsilon2 = 1.0 - epsilon1
    lambda_val_1 = lambda_coeff_1 / N
    lambda_val_2 = lambda_coeff_2 / N

    w_exp_rapide = epsilon1 * np.exp(-lambda_val_1 * t_centered)
    w_exp_lente = epsilon2 * np.exp(-lambda_val_2 * t_centered)
    w_reg = w_hann * (w_exp_rapide + w_exp_lente)

    # Simulation de l'OLA (fenêtres au carré)
    total_length = 5 * N
    ola_buffer = np.zeros(total_length)
    for hop_step in range(0, total_length - N, n_hop):
        ola_buffer[hop_step:hop_step + N] += w_reg ** 2

    # Zone centrale stable (élimine les effets de bord)
    stable_zone = ola_buffer[2 * N:3 * N]
    mean_ola = np.mean(stable_zone)
    peak_to_peak = np.max(stable_zone) - np.min(stable_zone)
    relative_error_db = 20 * np.log10(peak_to_peak / mean_ola + 1e-12)

    return relative_error_db


# ======================================================================
# 3. Balayage complet
# ======================================================================

def run_sweep(
    N: int = 512,
    hop_factors=None,
    alpha_ola_values=None,
    beta_grad: float = 1.0,
    output_path: str = "sweep_results.json",
):
    if hop_factors is None:
        hop_factors = ResultsStore.DEFAULT_HOP_FACTORS
    if alpha_ola_values is None:
        alpha_ola_values = ResultsStore.DEFAULT_ALPHA_OLA_VALUES

    store = ResultsStore()
    total = len(hop_factors) * len(alpha_ola_values)
    done = 0

    print(f"Balayage de {len(hop_factors)} hop_factors × {len(alpha_ola_values)} alpha_ola = {total} configurations")
    print(f"  hop_factors  = {hop_factors}")
    print(f"  alpha_ola    = {alpha_ola_values}")
    print(f"  N            = {N}")
    print(f"  beta_grad    = {beta_grad}")
    print(f"  Grille (Log) = 20 × 50 × 40 = {20*50*40} candidats par config")
    print()

    for hf in hop_factors:
        for alpha in alpha_ola_values:
            done += 1
            nhop = N // hf
            print(f"[{done}/{total}] hop_factor={hf} (nhop={nhop}), alpha_ola={alpha:.0f} ... ", end="", flush=True)

            t0 = time.time()

            # Étape 1 : recherche de paramètres optimaux
            eps1, l1_c, l2_c, loss = find_optimal_params(
                N=N, hop_factor=hf, alpha_ola=alpha, beta_grad=beta_grad,
            )

            # Étape 2 : calcul de l'erreur relative de modulation OLA
            rel_err_db = compute_ola_relative_error(
                N=N, hop_factor=hf,
                epsilon1=eps1, lambda_coeff_1=l1_c, lambda_coeff_2=l2_c,
            )

            dt = time.time() - t0

            store.set(
                hop_factor=hf, alpha_ola=alpha,
                epsilon1=eps1, lambda_coeff_1=l1_c, lambda_coeff_2=l2_c,
                relative_error_db=rel_err_db,
            )

            print(f"eps1={eps1:.4f}, λ1={l1_c:.4f}/N, λ2={l2_c:.4f}/N, "
                  f"err={rel_err_db:.1f} dB, smoothness_loss={loss:.1f}  ({dt:.1f}s)")

    # Sauvegarde
    store.save(output_path)
    print(f"\nResultats sauvegardes dans {output_path}")

    # Tableau récapitulatif
    print("\n" + "=" * 80)
    store.summary(N=N)

    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balayage des parametres de fenetre regularisee")
    parser.add_argument("--N", type=int, default=683, help="Taille de fenetre ")
    parser.add_argument("--hop-factors", type=int, nargs="+", default=None,
                        help="Liste de hop_factors (default: 20 28 36 40)")
    parser.add_argument("--alpha-ola", type=float, nargs="+", default=None,
                        help="Liste de alpha_ola (default: 10e4 14e4 18e4 20e4)")
    parser.add_argument("--beta-grad", type=float, default=1.0,
                        help="Poids de la penalite spectrale (default: 1.0)")
    parser.add_argument("--output", type=str, default="/home/adubois/openunmix/OpenUnmix/eval_model/sweep_results.json",
                        help="Fichier de sortie JSON (default: sweep_results.json)")
    args = parser.parse_args()

    run_sweep(
        N=args.N,
        hop_factors=args.hop_factors,
        alpha_ola_values=args.alpha_ola,
        beta_grad=args.beta_grad,
        output_path=args.output,
    )