#!/usr/bin/env python3
"""
plot_single_config_fit.py
=========================
Generates a publication-quality figure showing the 3-state SSM rational fraction 
fit (sum of 3 complex low-pass filters) for a single regularized window configuration.
Saves the figure in the project's figure folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Window and Spectrum Computation
# ---------------------------------------------------------------------------

def compute_window_and_spectrum(N, eps1, l1_c, l2_c):
    """Computes w_reg = w_hann * (eps1*exp(-l1*|t|) + eps2*exp(-l2*|t|))
    and its magnitude spectrum via FFT with high zero-padding resolution."""
    t = np.arange(N)
    w_hann = np.hanning(N)
    t_centered = np.abs(t - N // 2)

    eps2 = 1.0 - eps1
    l1 = l1_c / N
    l2 = l2_c / N

    w_exp_rapide = eps1 * np.exp(-l1 * t_centered)
    w_exp_lente  = eps2 * np.exp(-l2 * t_centered)
    w_reg = w_hann * (w_exp_rapide + w_exp_lente)

    # High frequency resolution zero-padding
    N_fft = max(32768, 32 * N)
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft))
    W_reg = np.fft.fftshift(np.fft.fft(w_reg, N_fft))
    W_reg_mag_linear = np.abs(W_reg)

    return w_reg, W_reg_mag_linear, freqs


# ---------------------------------------------------------------------------
# 2. Fitting Helpers (2-state and 3-state models)
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


def _prefit_2states(w_fit, target_norm, target_db, weights, n_starts):
    ll, ld = _build_loss_fns(_rational_2states, w_fit, target_norm, target_db, weights)
    b = _bounds_2states()
    best_loss = np.inf;  best_x = None
    for x0 in _x0_list_2states(n_starts):
        r = _two_phase_optimize(ll, ld, x0, b)
        if r.fun < best_loss:
            best_loss = r.fun;  best_x = r.x
    return best_x


def fit_rational_fraction_3states(W_mag_target_linear, freqs, n_starts=10):
    """
    Fits H3(jw) = sum_i ci/(mu_i - jw) (3 complex poles) to target magnitude spectrum.
    """
    w_fit       = freqs * 2.0 * np.pi
    norm_factor = np.max(W_mag_target_linear)
    target_norm = W_mag_target_linear / norm_factor
    target_db   = 20.0 * np.log10(target_norm + 1e-12)
    weights     = target_norm + 0.01

    # Stage 1: 2-state pre-fit for hierarchical initialization
    x2 = _prefit_2states(w_fit, target_norm, target_db, weights, n_starts)
    mu1_r = x2[4];  mu2_r = x2[6]
    mu3_r = float(np.sqrt(abs(mu1_r * mu2_r)))
    x0_hier = [x2[0], x2[1], x2[2], x2[3],
               1e-4, 0.0,
               x2[4], x2[5], x2[6], x2[7],
               mu3_r, 0.0]

    # Stage 2: 3-state fit
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
# 3. Main Plotting and Output Function
# ---------------------------------------------------------------------------

# Precomputed optimal parameters for specific window configurations
# Key: (eps1, l1_c, l2_c)
# Value: best_x parameters (12-element array) for the 3-state rational fraction model
PRECOMPUTED_FITS = {
    (0.053, 0.498, 0.437): [
        -0.005500147766920256, 0.0006445658571398884, -0.005315580976918233,
        0.0005927766504464783, 0.01081826656760707, -0.0012375469570947777,
        0.006334651111479089, 0.008352609219988977, 0.006304710491617774,
        -0.008906961010706161, 0.006696710197326846, -0.00011047789531459299
    ]
}


def generate_publication_plot(cfg, N=683, n_starts=10, fit_model=False):
    # Set publication plot style params
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,  # Increased by 4 (from original 10 to 14)
        'grid.alpha': 0.5,
        'grid.linestyle': ':'
    })

    print(f"Computing window and spectrum for config: {cfg}")
    w, mag, freqs = compute_window_and_spectrum(N, cfg['eps1'], cfg['l1_c'], cfg['l2_c'])

    norm_factor = np.max(mag)
    w_fit = freqs * 2.0 * np.pi

    cfg_key = (cfg['eps1'], cfg['l1_c'], cfg['l2_c'])
    if not fit_model and cfg_key in PRECOMPUTED_FITS:
        print("Using precomputed optimal 3-state model parameters...")
        best_x = np.array(PRECOMPUTED_FITS[cfg_key])
        fit = np.abs(_rational_3states(best_x, w_fit)) * norm_factor
        
        # Calculate loss (MSE in dB) for verification
        target_norm = mag / norm_factor
        target_db = 20.0 * np.log10(target_norm + 1e-12)
        weights = target_norm + 0.01
        _, ld3 = _build_loss_fns(_rational_3states, w_fit, target_norm, target_db, weights)
        loss = ld3(best_x)
        print(f"Precomputed fit loaded. Loss (MSE in dB): {loss:.6f}")
    else:
        print("Fitting 3-state complex rational model (optimization)...")
        loss, best_x, fit = fit_rational_fraction_3states(mag, freqs, n_starts)
        print(f"Fit completed successfully. Loss (MSE in dB/lin): {loss:.6f}")
        print(f"FITTED_BEST_X: {list(best_x)}")

    mag_db = 20.0 * np.log10(mag + 1e-12)
    fit_db = 20.0 * np.log10(fit + 1e-12)
    ymax   = np.max(mag_db)

    # Normalize the upper bound of the magnitude to 0dB
    mag_db = mag_db - ymax
    fit_db = fit_db - ymax
    ymax   = 0.0

    # Calculate Overlap-Add (OLA) constraint validation
    n_hop = N // 20
    total_length = 5 * N
    ola_buffer = np.zeros(total_length)
    for hop_step in range(0, total_length - N, n_hop):
        ola_buffer[hop_step:hop_step + N] += w**2

    # Isolate stable central zone to remove border/activation transients
    stable_zone = ola_buffer[2*N:3*N]
    mean_ola = np.mean(stable_zone)
    peak_to_peak_fluctuation = np.max(stable_zone) - np.min(stable_zone)
    relative_error_db = 20 * np.log10(peak_to_peak_fluctuation / mean_ola + 1e-12)

    # Zoom in on a few periods of the stable zone (e.g., 4 overlap periods)
    zoom_len = 4 * n_hop
    t_ola = np.arange(zoom_len)
    normalized_ola_zoom = stable_zone[:zoom_len] / mean_ola

    # Create fig1 (Spectral fit: Full Spectrum)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 5.5))

    # Plot 1: Full Spectrum
    ax1.plot(freqs, mag_db, color='steelblue', alpha=0.85, lw=2.0, label='Target Window')
    ax1.plot(freqs, fit_db, color='firebrick', lw=1.5, ls='--', label='3-pole Rational Fit')
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_ylim([ymax - 110, ymax + 5])
    ax1.set_xlabel('Normalized Frequency')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    ax1.grid(True)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)

    fig1.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.26)
    
    fig_dir = Path("/home/adubois/openunmix/OpenUnmix/fig")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path_spec = fig_dir / "regularized_window_spectral_fit.png"
    fig1.savefig(out_path_spec, dpi=200)
    plt.close(fig1)
    print(f"[SUCCESS] Spectral figure saved at: {out_path_spec}")

    # Create fig2 (OLA Zoom)
    fig2, ax3 = plt.subplots(1, 1, figsize=(6.5, 5.5))

    # Plot 3: OLA Zoom (oscillations around 1.0)
    ax3.plot(t_ola, normalized_ola_zoom, color='purple', lw=2.0, label=r'$s_{OLA}$')
    ax3.axhline(1.0, color='gray', linestyle='--', lw=1.5, label='Target')
    ax3.set_xlabel('Samples (stable zone)')
    ax3.set_ylabel('Normalized Amplitude')
    ax3.grid(True)
    
    # Add text box for relative error
    ax3.text(0.05, 0.95, f"Relative Fluctuation:\n{relative_error_db:.1f} dB", 
             transform=ax3.transAxes, verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)

    fig2.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.26)
    
    out_path_ola = fig_dir / "regularized_window_ola_zoom.png"
    fig2.savefig(out_path_ola, dpi=200)
    plt.close(fig2)
    print(f"[SUCCESS] OLA figure saved at: {out_path_ola}")


if __name__ == "__main__":
    # Using the standard regularized window config: eps1=0.053, l1_c=0.498, l2_c=0.437 (OLA at -56dB)
    target_config = {'eps1': 0.053, 'l1_c': 0.498, 'l2_c': 0.437}
    # Set fit_model=True if you want to redo the optimization, otherwise uses precomputed values
    generate_publication_plot(target_config, fit_model=False)
