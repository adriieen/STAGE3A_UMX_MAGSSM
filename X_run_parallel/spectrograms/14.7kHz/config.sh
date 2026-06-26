#!/usr/bin/env bash
# ============================================================
#  CONFIG CENTRALE — modifier ICI, puis lancer generate_ranks.sh
# ============================================================

# --- Fine-tuning / Reprise ---
# Laisser vide pour entraînement from scratch
MODEL=""        # ex: "/path/to/model_dir"  → active --model (fine-tuning)
CHECKPOINT=""   # ex: "/path/to/checkpoint" → active --checkpoint (reprise)

# --- Topologie DDP ---
NNODES=6
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.76"
MASTER_PORT=12355

# Noms des machines (rank 0 en premier — sera renommé rank0_<RANK0_NAME>.sh)
# Longueur doit être égale à NNODES
RANK_NAMES=("nissan" "1" "2" "3" "4" "5")

# --- Script d'entraînement ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_spectrogram_parallel.py"

# --- Arguments du modèle / dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/n_states=1nbins/512bins_alpha5e0"
TARGET="vocals"
EPOCHS=150
BATCH_SIZE=8
NB_WORKERS=5
SEQ_DUR=4
CHUNK_DUR=1
NB_MAGSSM_STATES=682
N_FFT=682
N_HOP=102

# --- Régularisation L2 sur les parties imaginaires des valeurs propres ---
ALPHA=5      # facteur multiplicatif de la loss L2 (0 = pas de régularisation)
BETA=1          # offset sur la pénalité au-dessus de pi (0 = bord franc)

# --- Paramètres de stabilité / initialisation du SSM ---
EPS_STABILITY=0     # marge de stabilité eps pour Progressive_SSM (0 = tighter)
DT_MIN=0.001        # timescale minimale du log-step (défaut 1e-3)
DT_MAX=0.1          # timescale maximale du log-step (défaut 0.1)


# --- Flags booléens (mettre à 1 pour activer, 0 pour désactiver) ---
FLAG_IS_WAV=1
FLAG_MEL=1
FLAG_AMP=0
FLAG_OG=0           # initialisation originale MagSSM (B orthogonale, valeurs propres linéaires)
FLAG_REGUL_WINDOW=0 # regularize window with bilateral double-exponential envelope (1 to enable)
EPSILON1=0.07
LAMBDA_COEFF_1=0.77
LAMBDA_COEFF_2=0.85
