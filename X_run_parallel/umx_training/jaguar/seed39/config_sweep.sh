#!/usr/bin/env bash
# ============================================================
#  CONFIG CENTRALE SWEEP — modifier ICI, puis lancer generate_ranks_sweep.sh
# ============================================================

# --- Fine-tuning / Reprise ---
# Laisser vide pour entraînement from scratch
MODEL=""        # ex: "/path/to/model_dir"  → active --model (fine-tuning)
CHECKPOINT=""   # ex: "/path/to/checkpoint" → active --checkpoint (reprise)

# --- Topologie DDP ---
NNODES=6
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.72"
MASTER_PORT=12355

# Noms des machines (rank 0 en premier)
# Longueur doit être égale à NNODES
RANK_NAMES=("jaguar" "1" "2" "3" "4" "5")

# --- Script d'entraînement ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_parallel.py"

# --- Arguments du modèle / dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/umx_training/seed=39"
TARGET="vocals"
EPOCHS=150
BATCH_SIZE=16
NB_WORKERS=24
SEQ_DUR=4
N_FFT=682
N_HOP=34
HIDDEN_SIZE=170
SEED=39

# --- Flags booléens (mettre à 1 pour activer, 0 pour désactiver) ---
FLAG_IS_WAV=1
FLAG_USE_EDGE=0
FLAG_AMP=0

# ─────────────────────────────────────────────────────────────────────
# Paramètres de fenêtre à balayer
# Format : "epsilon1 lambda_coeff_1 lambda_coeff_2"
# ─────────────────────────────────────────────────────────────────────
WINDOW_CONFIGS=(
    # === Baseline : pas de régularisation ===
    # "0 0 0"

    # # === Configurations à balayer ===
    # "0.5000  15.0000   4.0000"  # -19.5dB        
    # "0.5000   8.0658   4.0000" # -25.5dB        
    # "0.5000   4.7027   4.0000" # -30dB          x
    # "0.0500   4.0000   2.9419 " #-35dB          x
    # "0.1000   4.0000   1.7152"  # -40B          x
    # "0.2      2.1694   1"       # -45dB         x
    # "0.0500   4.0000   0.7382" # -48.8dB        x
    # "0.0530   0.4980   0.4365" # -56dB
    "0.1580   0.1796    0.2572"  # -61dB
    "0.1055   0.1000   0.1291" # -67dB
)
