"""
plot_lambda_init.py — Distribution des valeurs propres du SSM à l'initialisation.

Instancie SedgeMask directement (sans charger de checkpoint) et sauvegarde
un graphique Plotly interactif (HTML) de la distribution de Λd = Λ·Δ :

  - Subplot 1 : Histogramme de Re(Λd)  — taux d'amortissement
  - Subplot 2 : ECDF de |Im(Λd)|  en log-scale  — fréquences
  - Subplot 3 : Scatter complexe Re(Λd) vs Im(Λd)

Usage :
    cd open-unmix-pytorch/openunmix/
    python plot_lambda_init.py --nb-magssm-states 256 --hidden-size 256 --mel
    python plot_lambda_init.py --nb-magssm-states 512 --hidden-size 512 --mel --chunk-dur 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Environnement
# ---------------------------------------------------------------------------
from path_config import setup_paths
setup_paths()

import sedge_mask  # noqa — doit être importé depuis openunmix/


# ---------------------------------------------------------------------------
# Construction du modèle vierge (initialisation aléatoire standard)
# ---------------------------------------------------------------------------

def build_fresh_model(args, device):
    """Instancie SedgeMask avec les mêmes arguments que train_mask.py."""
    import transforms
    import model as model_module

    n_fft      = args.nfft
    n_hop      = args.nhop
    nb_ch      = args.nb_channels
    sample_rate = args.sample_rate

    stft, _ = transforms.make_filterbanks(
        n_fft=n_fft, n_hop=n_hop, sample_rate=sample_rate
    )
    encoder = torch.nn.Sequential(
        stft, model_module.ComplexNorm(mono=nb_ch == 1)
    ).to(device)

    chunk_dur_frames = int(args.chunk_dur * sample_rate)

    unmix = sedge_mask.SedgeMask(
        nb_bins=n_fft // 2 + 1,
        nb_channels=nb_ch,
        hidden_size=args.hidden_size,
        nb_layers=args.nb_layers,
        dim_state=args.nb_magssm_states,
        d_out=args.nb_magssm_states,
        n_fft=n_fft,
        n_hop=n_hop,
        device=device,
        encoder=encoder,
        use_edge=args.use_edge,
        unidirectional=args.unidirectional,
        chunk_duration=chunk_dur_frames,
        log_distributed_frequencies=args.mel,
    ).to(device)

    return unmix


# ---------------------------------------------------------------------------
# Extraction des paramètres Λ, log_step, step_scale
# ---------------------------------------------------------------------------

def extract_lambda_d(model):
    """
    Retourne (re, im) des produits Λd = Λ·Δ pour magssm_encoder.mimo.seq.

    re  = Re(Λd)  → taux d'amortissement par pas de temps (doit être < 0)
    im  = Im(Λd)  → fréquence d'oscillation en rad/sample
    """
    try:
        ssm = model.magssm_encoder.mimo.seq
    except AttributeError as e:
        raise RuntimeError(
            "Impossible d'accéder à model.magssm_encoder.mimo.seq. "
            "Vérifie que use_edge=False ou que le SSM est bien initialisé."
        ) from e

    with torch.no_grad():
        Lambda   = ssm.Lambda.detach().float()           # [N, 2]
        log_step = ssm.log_step.detach().float()         # [N]
        step     = ssm.step_scale * torch.exp(log_step)  # Δ  [N]

        Lambda_c = torch.complex(Lambda[:, 0], Lambda[:, 1])
        Ld       = Lambda_c * step   # Λ·Δ  [N] complexe

    re  = Ld.real.numpy()
    im  = Ld.imag.numpy()
    return re, im


# ---------------------------------------------------------------------------
# Graphique Plotly interactif
# ---------------------------------------------------------------------------

def plot_lambda_distribution(re, im, save_path="./eigenvalues_init.html",
                              title_suffix="initialisation"):
    """
    3 subplots Plotly interactifs :
      1. Histogramme Re(Λd)
      2. ECDF |Im(Λd)| en log-scale
      3. Scatter complexe Re(Λd) vs Im(Λd)
    """
    N = len(re)
    mag = np.abs(re + 1j * im)

    # --- Stats console ---
    print(f"\n  Λd = Λ·Δ   ({N} états SSM)  —  {title_suffix}")
    print(f"    Re : mean={re.mean():.4e}  std={re.std():.4e}  "
          f"min={re.min():.4e}  max={re.max():.4e}")
    print(f"    Im : mean={im.mean():.4e}  std={im.std():.4e}  "
          f"min={im.min():.4e}  max={im.max():.4e}")
    print(f"    Re > 0 (instables) : {int((re > 0).sum())}/{N}")
    print(f"    |Im| < 1e-5        : {int((np.abs(im) < 1e-5).sum())}/{N}")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Re(Λd) — taux d'amortissement  (doit être < 0)",
            "|Im(Λd)| — fréquence en rad/sample  [ECDF, log-scale]",
            "Scatter complexe : Re(Λd) vs Im(Λd)",
        ),
        vertical_spacing=0.10,
    )

    # ---- Subplot 1 : Histogramme Re(Λd) ----
    re_valid = re[re > -50]   # on ignore les outliers très négatifs
    print(re_valid)
    fig.add_trace(
        go.Histogram(
            x=re_valid,
            nbinsx=100,
            marker_color="#4C72B0",
            opacity=0.85,
            name="Re(Λd)",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="crimson",
                  annotation_text="Re=0", row=1, col=1)
    fig.add_vline(x=re_valid.mean(), line_dash="solid", line_color="orange",
                  annotation_text=f"mean={re_valid.mean():.3e}", row=1, col=1)
    fig.update_xaxes(title_text="Re(Λd)", row=1, col=1)
    fig.update_yaxes(title_text="Nombre d'états", row=1, col=1)

    # ---- Subplot 2 : ECDF |Im(Λd)| log-scale ----
    abs_im = np.abs(im[im != 0])
    abs_im_sorted = np.sort(abs_im)
    ecdf_y = np.arange(1, len(abs_im_sorted) + 1) / len(abs_im_sorted)

    fig.add_trace(
        go.Scatter(
            x=abs_im_sorted,
            y=ecdf_y,
            mode="lines",
            line=dict(color="#55A868", width=2),
            name="ECDF |Im(Λd)|",
            showlegend=False,
            hovertemplate="|Im|=%{x:.3e}<br>F(x)=%{y:.4f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="|Im(Λd)|", type="log", row=2, col=1)
    fig.update_yaxes(title_text="F(x)  [ECDF]", row=2, col=1)

    for pct in [50, 90]:
        val = np.percentile(abs_im_sorted, pct)
        fig.add_vline(
            x=val, line_dash="dot", line_color="orange", opacity=0.7,
            annotation_text=f"P{pct}={val:.1e}",
            row=2, col=1,
        )

    # ---- Subplot 3 : Scatter complexe ----
    fig.add_trace(
        go.Scatter(
            x=re,
            y=im,
            mode="markers",
            marker=dict(
                size=3,
                color=re,
                colorscale="RdBu_r",
                colorbar=dict(title="Re(Λd)", thickness=12, x=1.02),
                showscale=True,
                opacity=0.6,
            ),
            name="états",
            showlegend=False,
            hovertemplate="Re=%{x:.3e}<br>Im=%{y:.3e}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="crimson",
                  opacity=0.5, row=3, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="crimson",
                  opacity=0.5, row=3, col=1)
    fig.update_xaxes(title_text="Re(Λd)", row=3, col=1)
    fig.update_yaxes(title_text="Im(Λd)", row=3, col=1)

    # ---- Mise en page globale ----
    fig.update_layout(
        height=1000,
        title=dict(
            text=(f"Distribution de Λd = Λ·Δ  —  {N} états SSM  "
                  f"[{title_suffix}]<br>"
                  f"<sup>Re>0 (instables): {int((re > 0).sum())}  |  "
                  f"|Im|<1e-5: {int((np.abs(im) < 1e-5).sum())}</sup>"),
            font=dict(size=14),
        ),
        template="plotly_white",
        font=dict(family="Arial", size=11),
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"\n  ✓ Graphique interactif sauvegardé → {save_path}")
    print(f"    (ouvrir dans un navigateur pour zoomer)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution de Λd à l'initialisation du SSM."
    )
    # Paramètres du modèle (mêmes noms que train_mask.py)
    parser.add_argument("--nb-magssm-states", type=int, default=256,
                        dest="nb_magssm_states")
    parser.add_argument("--hidden-size", type=int, default=256,
                        dest="hidden_size")
    parser.add_argument("--nb-channels", type=int, default=2,
                        dest="nb_channels")
    parser.add_argument("--nb-layers", type=int, default=3,
                        dest="nb_layers")
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--nhop", type=int, default=1024)
    parser.add_argument("--chunk-dur", type=float, default=1.0,
                        dest="chunk_dur")
    parser.add_argument("--sample-rate", type=float, default=44100.0,
                        dest="sample_rate")
    parser.add_argument("--mel", action="store_true",
                        help="Log-distributed frequencies (comme --mel dans train_mask)")
    parser.add_argument("--use-edge", action="store_true", dest="use_edge")
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--output", type=str, default="./eigenvalues_init.html",
                        help="Chemin de sortie du fichier HTML")
    parser.add_argument("--seed", type=int, default=None,
                        help="Graine aléatoire pour reproductibilité de l'init")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"  Graine fixée : {args.seed}")

    device = torch.device(args.device)

    print("\n  Instanciation du modèle (initialisation fraîche)...")
    model = build_fresh_model(args, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres entraînables : {total_params:,}")

    re, im = extract_lambda_d(model)

    save_path = args.output
    if not save_path.endswith(".html"):
        save_path = save_path.replace(".png", ".html")
        if not save_path.endswith(".html"):
            save_path += ".html"

    plot_lambda_distribution(
        re, im,
        save_path=save_path,
        title_suffix=f"init  |  N={args.nb_magssm_states}  H={args.hidden_size}  mel={args.mel}",
    )


if __name__ == "__main__":
    main()
