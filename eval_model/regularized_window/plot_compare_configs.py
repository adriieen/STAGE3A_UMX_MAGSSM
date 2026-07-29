#!/usr/bin/env python3
"""
compare_configs.py
==================
Outil de diagnostic pour comparer deux configurations de fenetre hybride
(Hann x exp) via un fit SSM 3 etats.

- N_fft = max(32768, 32*N)  : resolution maximale du lobe principal
- Multistart log-uniforme sur Re(mu) in [1e-4, 1]
- Initialisation hierarchique interne (pre-fit 2 etats -> init c3=0)
- Visualisation : spectre complet + zoom lobe principal pour chaque config
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. Calcul de la fenetre et de son spectre
# ---------------------------------------------------------------------------

def compute_window_and_spectrum(N, eps1, l1_c, l2_c):
    """Construit w_reg = w_hann * (eps1*exp(-l1*|t|) + eps2*exp(-l2*|t|))
    et son spectre magnitude par FFT avec zero-padding maximal."""
    t = np.arange(N)
    w_hann = np.hanning(N)
    t_centered = np.abs(t - N // 2)

    eps2 = 1.0 - eps1
    l1 = l1_c / N
    l2 = l2_c / N

    w_exp_rapide = eps1 * np.exp(-l1 * t_centered)
    w_exp_lente  = eps2 * np.exp(-l2 * t_centered)
    w_reg = w_hann * (w_exp_rapide + w_exp_lente)

    # Zero-padding maximal : resolution frequentielle ~1/(32N)
    # Pour N=682 -> Df ~ 4.6e-5 -> ~87 points dans le lobe principal
    N_fft = max(32768, 32 * N)
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft))
    W_reg = np.fft.fftshift(np.fft.fft(w_reg, N_fft))
    W_reg_mag_linear = np.abs(W_reg)

    return w_reg, W_reg_mag_linear, freqs


# ---------------------------------------------------------------------------
# 2. Helpers internes (modeles 2 et 3 etats)
#    Le modele 2 etats est PRIVE : utilise uniquement pour l'init hierarchique
# ---------------------------------------------------------------------------

def _rational_2states(x, w_fit):
    c1  = x[0] + 1j * x[1];  c2  = x[2] + 1j * x[3]
    mu1 = x[4] + 1j * x[5];  mu2 = x[6] + 1j * x[7]
    return c1 / (mu1 - 1j * w_fit) + c2 / (mu2 - 1j * w_fit)


def _rational_3states(x, w_fit):
    c1  = x[0]  + 1j * x[1];  c2  = x[2]  + 1j * x[3];  c3  = x[4]  + 1j * x[5]
    mu1 = x[6]  + 1j * x[7];  mu2 = x[8]  + 1j * x[9];  mu3 = x[10] + 1j * x[11]
    return (  c1 / (mu1 - 1j * w_fit)
            + c2 / (mu2 - 1j * w_fit)
            + c3 / (mu3 - 1j * w_fit))


def _bounds_2states():
    return [(None,None),(None,None),(None,None),(None,None),
            (1e-6,None),(None,None),(1e-6,None),(None,None)]


def _bounds_3states():
    return [(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),
            (1e-6,None),(None,None),(1e-6,None),(None,None),(1e-6,None),(None,None)]


def _build_loss_fns(rational_fn, w_fit, target_norm, target_db, weights):
    def loss_linear(x):
        H = rational_fn(x, w_fit)
        return np.mean(weights * (np.abs(H) - target_norm) ** 2)
    def loss_db(x):
        H = rational_fn(x, w_fit)
        pred_db = 20.0 * np.log10(np.abs(H) + 1e-12)
        return np.mean(weights * (pred_db - target_db) ** 2)
    return loss_linear, loss_db


def _two_phase_optimize(loss_lin, loss_db, x0, bounds):
    """Phase 1 : lineaire (pre-conditionnement). Phase 2 : dB (convergence fine)."""
    res = minimize(loss_lin, x0, bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    res = minimize(loss_db, res.x, bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8})
    return res


def _x0_list_2states(n_starts, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    grid = np.logspace(-4, 0, n_starts)
    return [[1.0, 0.0, 0.1, 0.0, mu_r, rng.uniform(-mu_r, mu_r), mu_r * 10, 0.0]
            for mu_r in grid]


def _x0_list_3states(n_starts, x0_hier=None, rng_seed=42):
    """c3 quasi-nul dans l'init hierarchique : evite les interferences destructives."""
    rng = np.random.default_rng(rng_seed)
    grid = np.logspace(-4, 0, n_starts)
    cands = []
    if x0_hier is not None:
        cands.append(x0_hier)
    for mu_r in grid:
        mu_i = rng.uniform(-mu_r, mu_r)
        cands.append([1.0, 0.0, 0.1, 0.0, 1e-4, 0.0,
                      mu_r, mu_i, mu_r * 10, mu_i, mu_r * 3, 0.0])
    return cands


# ---------------------------------------------------------------------------
# 3. Pre-fit 2 etats interne (pour l'init hierarchique uniquement)
# ---------------------------------------------------------------------------

def _prefit_2states(w_fit, target_norm, target_db, weights, n_starts):
    """Retourne x*_2 (vecteur optimal 2 etats). Usage : init hierarchique du 3 etats."""
    ll, ld = _build_loss_fns(_rational_2states, w_fit, target_norm, target_db, weights)
    b = _bounds_2states()
    best_loss = np.inf;  best_x = None
    for x0 in _x0_list_2states(n_starts):
        r = _two_phase_optimize(ll, ld, x0, b)
        if r.fun < best_loss:
            best_loss = r.fun;  best_x = r.x
    return best_x


# ---------------------------------------------------------------------------
# 4. Fit SSM 3 etats (interface publique)
# ---------------------------------------------------------------------------

def fit_rational_fraction_3states(W_mag_target_linear, freqs, n_starts=5):
    """
    Ajuste H3(jw) = sum_i ci/(mu_i - jw)  (3 poles complexes).

    Protocole :
      Etape 1 — Pre-fit 2 etats interne (n_starts conditions)
                pour construire l'init hierarchique [x*2, c3=1e-4, mu3=geometrique].
      Etape 2 — Fit 3 etats : init hierarchique + n_starts conditions independantes.
      Selection — Minimum global parmi tous les runs.

    Retourne : (loss_finale, x_optimal, H_mag_lineaire)
    """
    w_fit       = freqs * 2.0 * np.pi
    norm_factor = np.max(W_mag_target_linear)
    target_norm = W_mag_target_linear / norm_factor
    target_db   = 20.0 * np.log10(target_norm + 1e-12)
    weights     = target_norm + 0.01

    # -- Etape 1 : pre-fit 2 etats -> init hierarchique -------------------
    x2 = _prefit_2states(w_fit, target_norm, target_db, weights, n_starts)
    mu1_r = x2[4];  mu2_r = x2[6]
    mu3_r = float(np.sqrt(abs(mu1_r * mu2_r)))
    x0_hier = [x2[0], x2[1], x2[2], x2[3],
               1e-4, 0.0,          # c3 quasi-nul : evite les interferences
               x2[4], x2[5], x2[6], x2[7],
               mu3_r, 0.0]

    # -- Etape 2 : fit 3 etats avec multistart ----------------------------
    ll3, ld3 = _build_loss_fns(_rational_3states, w_fit, target_norm, target_db, weights)
    b3 = _bounds_3states()
    best_loss = np.inf;  best_x = None
    for x0 in _x0_list_3states(n_starts, x0_hier=x0_hier):
        r = _two_phase_optimize(ll3, ld3, x0, b3)
        if r.fun < best_loss:
            best_loss = r.fun;  best_x = r.x

    H_full_mag = np.abs(_rational_3states(best_x, w_fit)) * norm_factor
    return best_loss, best_x, H_full_mag


# ---------------------------------------------------------------------------
# 5. Comparaison visuelle des deux configurations
# ---------------------------------------------------------------------------

def compare_configs_3states(cfg1, cfg2, N=683, n_starts=5):
    """
    Compare deux configurations (eps1, l1_c, l2_c) via fit SSM 3 etats.
    Produit un graphique 2x2 : spectre complet + zoom lobe principal.
    """

    def _process_config(label, cfg, freqs_ref=None):
        print(f"\n[{label}] Calcul du spectre (N_fft=max(32768,32*N))...")
        w, mag, freqs = compute_window_and_spectrum(N, cfg['eps1'], cfg['l1_c'], cfg['l2_c'])
        if freqs_ref is None:
            freqs_ref = freqs

        print(f"[{label}] Fit 3 etats — pre-fit 2 etats interne + multistart={n_starts}...")
        loss, x, fit, = fit_rational_fraction_3states(mag, freqs_ref, n_starts)
        print(f"  -> loss={loss:.6f}")

        return w, mag, freqs_ref, loss, fit

    # Traitement Config 1
    w1, mag1, freqs, loss1, fit1 = _process_config("Config 1", cfg1)

    # Traitement Config 2 (reutilise le meme vecteur freqs)
    w2, mag2, _,     loss2, fit2 = _process_config("Config 2", cfg2, freqs_ref=freqs)

    # -- Visualisation 2x2 ------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f'Fit SSM 3 etats  |  N={N}, N_fft=max(32768,32N), multistart={n_starts}',
        fontsize=11, fontweight='bold')
    t = np.arange(N)

    configs_data = [
        ('Config 1', w1, mag1, fit1, loss1, 'steelblue', 0),
        ('Config 2', w2, mag2, fit2, loss2, 'firebrick',  1),
    ]

    for label, w, mag, fit, loss, color, col in configs_data:
        mag_db = 20.0 * np.log10(mag + 1e-12)
        fit_db = 20.0 * np.log10(fit + 1e-12)
        ymax   = np.max(mag_db)

        # Ligne 0 : spectre complet
        ax = axs[0, col]
        ax.plot(freqs, mag_db, color=color, alpha=0.85, lw=1.5, label='Cible')
        ax.plot(freqs, fit_db, color='black', lw=1.5, ls='--',
                label=f'Fit 3 etats (loss={loss:.4f})')
        ax.set_title(f'{label} — Spectre complet', fontweight='bold')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([ymax - 120, ymax + 10])
        ax.set_xlabel('Frequence normalisee'); ax.grid(True)
        ax.legend(fontsize=9)

        # Ligne 1 : zoom lobe principal
        zoom_bw = max(4.0 / N, 0.01)
        ax_z = axs[1, col]
        ax_z.plot(freqs, mag_db, color=color, alpha=0.85, lw=1.5, label='Cible')
        ax_z.plot(freqs, fit_db, color='black', lw=1.5, ls='--',
                  label=f'Fit 3 etats (loss={loss:.4f})')
        ax_z.set_title(f'{label} — Zoom lobe principal (+/-{zoom_bw:.4f})')
        ax_z.set_xlim([-zoom_bw, zoom_bw])
        ax_z.set_ylim([ymax - 40, ymax + 3])
        ax_z.set_xlabel('Frequence normalisee'); ax_z.grid(True)
        ax_z.legend(fontsize=9)

    plt.tight_layout()
    out_path = 'comparison_ssm_fit_3states_sym_weighted.png'
    plt.savefig(out_path, dpi=150)
    print(f"\n[OK] Figure sauvegardee : {out_path}")

    print("\n" + "=" * 60)
    print("RESUME FINAL (SSM 3 etats)")
    print("=" * 60)
    print(f"Config 1 : loss={loss1:.6f}")
    print(f"Config 2 : loss={loss2:.6f}")
    print(f"Meilleur fit : {'Config 1' if loss1 < loss2 else 'Config 2'}")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    method1 ={'eps1': 0.1459,  'l1_c': 0.1351,  'l2_c': 0.4924}
    method2 = {'eps1': 0.053,  'l1_c': 0.498,  'l2_c': 0.437}
    compare_configs_3states(method1, method2)

# {'eps1': 0.04,  'l1_c': 0.1,  'l2_c': 0.4924}
# {'eps1': 0.1459,  'l1_c': 0.1351,  'l2_c': 0.4924}
# {'eps1': 0.053,  'l1_c': 0.498,  'l2_c': 0.437}