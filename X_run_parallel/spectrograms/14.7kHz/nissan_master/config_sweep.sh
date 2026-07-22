#!/usr/bin/env bash
# ============================================================
#  CONFIG CENTRALE SWEEP — modifier ICI, puis lancer generate_ranks_sweep.sh
# ============================================================

# --- Fine-tuning / Reprise ---
# Laisser vide pour entraînement from scratch
MODEL=""        # ex: "/path/to/model_dir"  → active --model (fine-tuning)
CHECKPOINT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/non_progressive_170bins_alpha4e-3_structured_init_bis"   # ex: "/path/to/checkpoint" → active --checkpoint (reprise)

# --- Topologie DDP ---
NNODES=6
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.76"
MASTER_PORT=12355

# Noms des machines (rank 0 en premier)
# Longueur doit être égale à NNODES
RANK_NAMES=("nissan" "1" "2" "3" "4" "5")

# --- Script d'entraînement ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_spectrogram_parallel.py"

# --- Arguments du modèle / dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/non_progressive_170bins_alpha4e-3_structured_init_bis"
TARGET="vocals"
EPOCHS=60
EPOCHS_RESUME=100    # Nombre d'époques supplémentaires à effectuer en cas de reprise
BATCH_SIZE=6
NB_WORKERS=5
SEQ_DUR=4
CHUNK_DUR=1
NB_MAGSSM_STATES=342
N_FFT=340
N_HOP=34
LEARNING_RATE=0.01

# --- Régularisation L2 sur les parties imaginaires des valeurs propres ---
ALPHA=0.004    # facteur multiplicatif de la loss L2 (0 = pas de régularisation)
BETA=0        # offset sur la pénalité au-dessus de pi (0 = bord franc)

# --- Paramètres de stabilité / initialisation du SSM ---
EPS_STABILITY=0     # marge de stabilité eps pour Progressive_SSM (0 = tighter)
DT_MIN=0.001        # timescale minimale du log-step (défaut 1e-3)
DT_MAX=0.1          # timescale maximale du log-step (défaut 0.1)

# --- Flags booléens (mettre à 1 pour activer, 0 pour désactiver) ---
FLAG_IS_WAV=1
FLAG_MEL=0
FLAG_AMP=0
FLAG_OG=1           # initialisation originale MagSSM
FLAG_COMPLEX_SPECTROGRAM=1  
FLAG_STRUCTURED_INITIALISATION=1 #linearly spaced imaginary parts of the eigenvalues between 0 and Pi

# ─────────────────────────────────────────────────────────────────────
# Paramètres de fenêtre à balayer
# Format : "epsilon1 lambda_coeff_1 lambda_coeff_2"
# ─────────────────────────────────────────────────────────────────────
WINDOW_CONFIGS=(
    # === Baseline : pas de régularisation ===
    # "0 0 0"

    # === Configurations à balayer ===
    "0.5000   4.7027   4.0000" # -30dB
    # "0.0500   4.0000   2.9419 " #-35dB
    # "0.1000   4.0000   1.7152"  # -40B
    # "0.2      2.1694   1"       # -45dB

    # "0.0530   0.4980   0.4365" # -56dB
#     "0.1580   0.1796    0.2572"  # -61dB
#     "0.1055   0.1000   0.1291" # -67dB
)
