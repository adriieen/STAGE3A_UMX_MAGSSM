#!/usr/bin/env bash
# ============================================================
#  CONFIG CENTRALE SWEEP DECODER — sacrum master
# ============================================================

# --- Fine-tuning / Reprise ---
MODEL=""        # ex: "/path/to/model_dir"  → active --model (fine-tuning)
CHECKPOINT=""   # ex: "/path/to/checkpoint" → active --checkpoint (reprise)

# --- Topologie DDP ---
NNODES=6
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.51"
MASTER_PORT=12355

# Noms des machines (rank 0 en premier)
RANK_NAMES=("sacrum" "1" "2" "3" "4" "5")

# --- Script d'entraînement ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_decoder_parallel.py"

# --- Arguments du modèle / dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_decoders/[341bins_modified_nhop_low_lr]spectrogram_loss_modified_ssm"
TARGET="vocals"
EPOCHS=20
EPOCHS_RESUME=0    # Nombre d'époques supplémentaires à effectuer en cas de reprise
BATCH_SIZE=6
NB_WORKERS=5
SEQ_DUR=4
CHUNK_DUR=1
NB_MAGSSM_STATES=341
N_FFT=680
N_HOP=68
LEARNING_RATE=0.001

# --- Régularisation L2 sur les parties imaginaires des valeurs propres ---
ALPHA=0    # facteur multiplicatif de la loss L2
BETA=0        # offset sur la pénalité au-dessus de pi

# --- Paramètres de stabilité / initialisation du SSM ---
EPS_STABILITY=0     # marge de stabilité eps pour Progressive_SSM
DT_MIN=0.1          # timescale minimale du log-step
DT_MAX=10.0         # timescale maximale du log-step

# --- Flags booléens (1 pour activer, 0 pour désactiver) ---
FLAG_IS_WAV=1
FLAG_PROGRESSIVE=0
FLAG_FFT_KERNEL=1
FLAG_MEL=0
FLAG_AMP=0
FLAG_OG=1           # initialisation originale MagSSM
FLAG_STRUCTURED_INITIALISATION=1 # linearly spaced imaginary parts of eigenvalues between 0 and Pi / O and Nbins if working in the frequency domain.
FLAG_PHASE_CORRECTION=0 
# ─────────────────────────────────────────────────────────────────────
# Paramètres de fenêtre à balayer
# Format : "epsilon1 lambda_coeff_1 lambda_coeff_2"
# ─────────────────────────────────────────────────────────────────────
WINDOW_CONFIGS=(
    "0.5000   4.7027   4.0000" # -30dB
)
