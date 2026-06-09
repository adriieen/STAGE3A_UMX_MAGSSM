"""
diagnose_nan.py — Diagnostic des NaN dans le checkpoint SedgeMask / SSM.

Usage (depuis le répertoire openunmix/) :
    python diagnose_nan.py --checkpoint /chemin/vers/dossier/checkpoint/

Que fait ce script :
  1. Charge le .chkpnt (état du réseau à la meilleure époque)
  2. Inspecte chaque paramètre : min, max, norme, ratio NaN/Inf
  3. Reconstruit Lambda_bar (valeurs propres discrétisées) et diagnostique
     la stabilité de chaque état SSM
  4. Fait un forward pass sur un signal synthétique et enregistre
     les statistiques à chaque couche via des hooks
  5. Produit un rapport textuel résumant les points critiques
"""

import sys
import os
import argparse
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non-interactif
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Chargement de l'environnement
# ---------------------------------------------------------------------------
from path_config import setup_paths
setup_paths()

import sedge_mask
import utils_edge_var

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(t, name=""):
    """Résumé compact d'un tenseur : min/max/mean/std/NaN/Inf."""
    t = t.float().detach()
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    n_tot = t.numel()
    tag = ""
    if n_nan > 0:
        tag += f" ⚠ NaN={n_nan}/{n_tot}"
    if n_inf > 0:
        tag += f" ⚠ Inf={n_inf}/{n_tot}"
    if n_nan == 0 and n_inf == 0:
        return (
            f"  min={t.min():.4e}  max={t.max():.4e}"
            f"  mean={t.mean():.4e}  std={t.std():.4e}"
            f"  norm={t.norm():.4e}{tag}"
        )
    else:
        return f"  {tag}"


def section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


# ---------------------------------------------------------------------------
# 1. Inspection des paramètres du checkpoint
# ---------------------------------------------------------------------------

def inspect_parameters(state_dict):
    section("PARAMÈTRES DU CHECKPOINT")
    problems = []
    for name, param in state_dict.items():
        p = param.float()
        n_nan = torch.isnan(p).sum().item()
        n_inf = torch.isinf(p).sum().item()
        line = f"  {name:60s} shape={list(p.shape)}"
        if n_nan > 0 or n_inf > 0:
            line += f"  *** NaN={n_nan} Inf={n_inf} ***"
            problems.append(name)
        else:
            line += fmt(p)
        print(line)
    if problems:
        print(f"\n  *** {len(problems)} paramètre(s) corrompus : ***")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  Aucun NaN/Inf dans les paramètres sauvegardés.")
    return problems


# ---------------------------------------------------------------------------
# 2. Diagnostic SSM : Lambda_bar (valeurs propres discrétisées)
# ---------------------------------------------------------------------------

def diagnose_lambda(model):
    section("DIAGNOSTIC SSM : VALEURS PROPRES DISCRÉTISÉES (Lambda_bar)")
    
    from model_edge.ssm_bis import Progressive_SSM
    from model_edge.ssm_bis import discretize_zoh, as_complex

    ssm_modules = [(name, m) for name, m in model.named_modules()
                   if isinstance(m, Progressive_SSM)]

    if not ssm_modules:
        print("  Aucun module Progressive_SSM trouvé dans le modèle.")
        return

    for name, ssm in ssm_modules:
        print(f"\n  Module : {name}")
        Lambda = ssm.Lambda.detach().float()
        log_step = ssm.log_step.detach().float()
        step = ssm.step_scale * torch.exp(log_step)

        # Reconstruction Lambda complexe
        Lambda_c = torch.complex(Lambda[:, 0], Lambda[:, 1])
        
        # Vérif Re(Lambda)
        real_parts = Lambda_c.real
        n_positive = (real_parts >= 0).sum().item()
        n_close_zero = (real_parts.abs() < 1e-3).sum().item()
        print(f"    Re(Lambda) : min={real_parts.min():.4e}  max={real_parts.max():.4e}")
        print(f"    États avec Re(Lambda) >= 0        : {n_positive}/{len(real_parts)}  ← instables!")
        print(f"    États avec |Re(Lambda)| < 1e-3    : {n_close_zero}/{len(real_parts)}  ← quasi-instables")

        # Lambda_bar = exp(Lambda * Delta)
        Lambda_bar = Lambda_c * step
        magnitudes = Lambda_bar.abs()
        n_gt1 = (magnitudes > 1.0).sum().item()
        n_near1 = ((magnitudes - 1.0).abs() < 1e-3).sum().item()
        print(f"    |Lambda_bar| : min={magnitudes.min():.4e}  max={magnitudes.max():.4e}  mean={magnitudes.mean():.4e}")
        print(f"    États instables |Lambda_bar| > 1  : {n_gt1}/{len(magnitudes)}  ← NaN garanti si > 0!")
        print(f"    États quasi-stables ||Lb|-1|<1e-3 : {n_near1}/{len(magnitudes)}")

        # Step (timescale)
        print(f"    Delta (step) : min={step.min():.4e}  max={step.max():.4e}  mean={step.mean():.4e}")

        # Top-5 états les plus proches de l'instabilité
        worst_idx = magnitudes.argsort(descending=True)[:5]
        print(f"    Top-5 états les plus proches de l'instabilité :")
        for i in worst_idx.tolist():
            print(f"      état {i:4d} : |Lambda_bar|={magnitudes[i]:.6f}"
                  f"  Re(Λ)={real_parts[i]:.4e}  Im(Λ)={Lambda_c.imag[i]:.4e}"
                  f"  Δ={step[i]:.4e}")


# ---------------------------------------------------------------------------
# 2b. Distribution des valeurs propres discrétisées (plot PNG)
# ---------------------------------------------------------------------------

def plot_lambda_distribution(model, save_path="./eigenvalues_trained.png"):
    """
    Calcule Lambda_bar = exp(Lambda * Delta) pour le magssm_encoder
    et sauvegarde un graphique 3x1 avec :
      - histogramme de Re(Lambda_bar)
      - histogramme de Im(Lambda_bar)
      - histogramme de |Lambda_bar|
    dans save_path.

    Chemin des paramètres :
        model.magssm_encoder.mimo.seq.Lambda    [N, 2]  (re, im)
        model.magssm_encoder.mimo.seq.log_step  [N]
        model.magssm_encoder.mimo.seq.step_scale  (scalaire)
    """
    try:
        ssm = model.magssm_encoder.mimo.seq   # Progressive_SSM
    except AttributeError:
        print("  ⚠ plot_lambda_distribution : magssm_encoder.mimo.seq introuvable.")
        return

    with torch.no_grad():
        Lambda   = ssm.Lambda.detach().float()      # [N, 2]
        log_step = ssm.log_step.detach().float()    # [N]
        step     = ssm.step_scale * torch.exp(log_step)  # Delta  [N]

        Lambda_c   = torch.complex(Lambda[:, 0], Lambda[:, 1])  # [N] complexe
        Lambda_bar = torch.exp(Lambda_c * step)                  # [N] complexe

        re  = Lambda_bar.real.numpy()
        im  = Lambda_bar.imag.numpy()
        mag = np.abs(re + 1j * im)

    N = len(re)
    n_unstable = int((mag > 1.0).sum())
    n_near1    = int((np.abs(mag - 1.0) < 1e-3).sum())

    fig, axes = plt.subplots(2, 1, figsize=(10, 11))
    fig.suptitle(
        f"Distribution de $\\Lambda_{{bar}} = \\exp(\\Lambda \\cdot \\Delta)$\n",
        # f"({N} états SSM  |  instables |Λ̄|>1 : {n_unstable}  |  quasi-stables : {n_near1})",
        fontsize=12, fontweight="bold"
    )

    # --- Subplot 1 : Re(Lambda_bar) ---
    ax = axes[0]
    ax.hist(re, bins=500, color="#4C72B0", edgecolor="none", alpha=0.85)
    ax.axvline(0,          color="crimson", linewidth=1.2, linestyle="--", label="Re=0")
    ax.axvline(re.mean(),  color="orange",  linewidth=1.2, linestyle="-",
               label=f"mean={re.mean():.4f}")
    ax.set_xlabel(r"$\mathrm{Re}(\Lambda_{\mathrm{bar}})$", fontsize=11)
    ax.set_ylabel("Nombre d'états", fontsize=10)
    ax.set_title(r"Partie réelle de $\Lambda_{\mathrm{bar}}$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2 : Im(Lambda_bar) ---
    ax = axes[1]
    ax.hist(im, bins=300, color="#55A868", edgecolor="none", alpha=0.85)
    ax.axvline(0,          color="crimson", linewidth=1.2, linestyle="--", label="Im=0")
    ax.axvline(im.mean(),  color="orange",  linewidth=1.2, linestyle="-",
               label=f"mean={im.mean():.4f}")
    ax.set_xlabel(r"$\mathrm{Im}(\Lambda_{\mathrm{bar}})$", fontsize=11)
    ax.set_ylabel("Nombre d'états", fontsize=10)
    ax.set_title(r"Partie imaginaire de $\Lambda_{\mathrm{bar}}$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)



    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Distribution Lambda_bar sauvegardée → {save_path}")
    print(f"    Re : mean={re.mean():.4e}  std={re.std():.4e}  min={re.min():.4e}  max={re.max():.4e}")
    print(f"    Im : mean={im.mean():.4e}  std={im.std():.4e}  min={im.min():.4e}  max={im.max():.4e}")



# ---------------------------------------------------------------------------
# 3. Forward pass avec hooks → stats par couche
# ---------------------------------------------------------------------------

def forward_diagnostic(model, device, nb_bins=2049, nb_channels=2,
                        seq_dur_s=2.0, sample_rate=44100, n_hop=1024):
    section("FORWARD PASS DIAGNOSTIQUE (signal synthétique Gaussien)")

    hooks = []
    layer_stats = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return
            o = out.detach().float()
            n_nan = torch.isnan(o).sum().item()
            n_inf = torch.isinf(o).sum().item()
            layer_stats[name] = {
                "shape": list(o.shape),
                "min": o.min().item() if n_nan == 0 and n_inf == 0 else float('nan'),
                "max": o.max().item() if n_nan == 0 and n_inf == 0 else float('nan'),
                "norm": o.norm().item() if n_nan == 0 and n_inf == 0 else float('nan'),
                "n_nan": n_nan,
                "n_inf": n_inf,
            }
        return hook

    for name, module in model.named_modules():
        if name:  # ignorer le module racine
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    T = int(seq_dur_s * sample_rate)
    x = torch.randn(1, nb_channels, T).to(device) * 0.1  # signal faible, réaliste

    with torch.no_grad():
        try:
            _ = model(x)
        except Exception as e:
            print(f"  ⚠ Exception pendant le forward : {e}")

    for h in hooks:
        h.remove()

    # Affichage : on ne montre que les couches problématiques et leurs voisines
    names = list(layer_stats.keys())
    problem_indices = [i for i, n in enumerate(names)
                       if layer_stats[n]['n_nan'] > 0 or layer_stats[n]['n_inf'] > 0]

    if not problem_indices:
        print("  ✓ Aucun NaN/Inf dans les activations. Forward pass propre.")
        # Afficher quand même les normes pour détecter une explosion progressive
        print("\n  Normes des activations (toutes couches) :")
        for name in names:
            s = layer_stats[name]
            print(f"    {name:60s}  norm={s['norm']:.4e}  shape={s['shape']}")
    else:
        # Identifier la première couche qui explose
        first_problem = problem_indices[0]
        print(f"  ✗ Première couche avec NaN/Inf : [{first_problem}] {names[first_problem]}")
        
        # Afficher les 3 couches avant + la couche problématique + celles d'après
        window = set()
        for idx in problem_indices:
            for j in range(max(0, idx-3), min(len(names), idx+4)):
                window.add(j)

        print("\n  Couches autour des problèmes :")
        prev_ok = None
        for i, name in enumerate(names):
            s = layer_stats[name]
            is_problem = s['n_nan'] > 0 or s['n_inf'] > 0
            if i in window:
                if prev_ok is not None and i - prev_ok > 1:
                    print(f"    {'...'}")
                tag = "  *** NaN/Inf ***" if is_problem else ""
                print(f"    [{i:3d}] {name:55s}  norm={s['norm']:.3e}  shape={s['shape']}{tag}")
                prev_ok = i

    return layer_stats, problem_indices


# ---------------------------------------------------------------------------
# 4. Résumé et recommandations
# ---------------------------------------------------------------------------

def print_summary(param_problems, layer_stats, problem_indices):
    section("RÉSUMÉ ET RECOMMANDATIONS")
    
    if param_problems:
        print(f"  • Les paramètres eux-mêmes contiennent des NaN/Inf.")
        print(f"    → Le checkpoint est corrompu (entraînement en NaN trop longtemps).")
        print(f"    → Recharger le .pth (best model) au lieu du .chkpnt.")
    
    names = list(layer_stats.keys())
    if problem_indices:
        first = names[problem_indices[0]]
        print(f"\n  • Première explosion : couche '{first}'")
        
        if "Lambda" in first or "ssm" in first.lower() or "progressive" in first.lower():
            print(f"    → Cause probable : Re(Lambda) ≥ 0, valeurs propres instables.")
            print(f"    → Fix : eps_stability dans ensure_stability (voir ssm_bis.py).")
        elif "magssm" in first.lower():
            print(f"    → Cause probable : explosion dans le MAGSSM avant le SSM.")
        elif "ln" in first.lower() or "norm" in first.lower():
            print(f"    → LayerNorm sur des activations déjà Inf → NaN.")
            print(f"    → La source est en amont, chercher la première couche non-Inf.")
        elif "fc" in first.lower():
            print(f"    → Linear avec des poids ou des entrées Inf/NaN.")
    else:
        print("  • Forward pass propre sur ce checkpoint.")
        print("    → L'explosion se produit pendant l'entraînement (gradient).")
        print("    → Piste : monitorer les gradients de Lambda avec register_backward_hook.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnostic NaN du modèle SedgeMask")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le dossier contenant vocals.chkpnt et separator.json")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device à utiliser (cpu ou cuda:0)")
    parser.add_argument("--seq-dur", type=float, default=2.0,
                        help="Durée du signal synthétique de test (secondes)")
    args = parser.parse_args()

    device = torch.device(args.device)
    chkpnt_dir = Path(args.checkpoint)

    # --- Charger le separator.json pour les params du modèle ---
    import json
    sep_path = chkpnt_dir / "separator.json"
    with open(sep_path) as f:
        sep_conf = json.load(f)

    vocals_path = chkpnt_dir / "vocals.json"
    with open(vocals_path) as f:
        vocals_conf = json.load(f)
    model_args = vocals_conf["args"]

    # --- Charger le checkpoint ---
    chkpnt_path = chkpnt_dir / "vocals.chkpnt"
    pth_path = chkpnt_dir / "vocals.pth"

    print(f"\nChargement du checkpoint : {chkpnt_path}")
    chkpnt = torch.load(chkpnt_path, map_location="cpu")
    state_dict = chkpnt["state_dict"]

    # 1. Inspection des paramètres
    param_problems = inspect_parameters(state_dict)

    # --- Reconstruire le modèle pour le forward pass ---
    print("\nReconstruction du modèle...")
    n_fft = model_args.get("nfft", 4096)
    n_hop = model_args.get("nhop", 1024)
    nb_channels = model_args.get("nb_channels", 2)
    hidden_size = model_args.get("hidden_size", 256)
    nb_layers = model_args.get("nb_layers", 3)
    dim_state = model_args.get("nb_magssm_states", 256)
    chunk_dur = model_args.get("chunk_dur", 1.0)
    sample_rate = model_args.get("sample_rate", 44100.0)
    mel = model_args.get("mel", False)

    import transforms, model as model_module
    stft, _ = transforms.make_filterbanks(
        n_fft=n_fft, n_hop=n_hop, sample_rate=sample_rate
    )
    encoder = torch.nn.Sequential(
        stft, model_module.ComplexNorm(mono=nb_channels == 1)
    ).to(device)

    unmix = sedge_mask.SedgeMask(
        nb_bins=n_fft // 2 + 1,
        nb_channels=nb_channels,
        hidden_size=hidden_size,
        nb_layers=nb_layers,
        dim_state=dim_state,
        d_out=dim_state,          # ← manquait : sans ça, default=129 → mismatch
        n_fft=n_fft,
        n_hop=n_hop,
        device=device,
        encoder=encoder,
        use_edge=model_args.get("use_edge", True),
        unidirectional=model_args.get("unidirectional", False),
        chunk_duration=int(chunk_dur * sample_rate),
        log_distributed_frequencies=mel,
    ).to(device)

    # Charger les poids (tolérant aux clés manquantes)
    missing, unexpected = unmix.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Clés manquantes dans le state_dict ({len(missing)}) : {missing[:5]}...")
    if unexpected:
        print(f"  Clés inattendues ({len(unexpected)}) : {unexpected[:5]}...")

    # 2. Diagnostic Lambda_bar + plot distribution
    # diagnose_lambda(unmix)
    plot_lambda_distribution(unmix, save_path="./eigenvalues_trained.png")

    # # 3. Forward pass avec hooks
    # layer_stats, problem_indices = forward_diagnostic(
    #     unmix, device,
    #     nb_bins=n_fft // 2 + 1,
    #     nb_channels=nb_channels,
    #     seq_dur_s=args.seq_dur,
    #     sample_rate=int(sample_rate),
    #     n_hop=n_hop,
    # )

    # # 4. Résumé
    # print_summary(param_problems, layer_stats, problem_indices)


if __name__ == "__main__":
    main()
