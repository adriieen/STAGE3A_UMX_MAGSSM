#!/usr/bin/env bash
# ============================================================
#  CONFIG CENTRALE SWEEP EVALUATION — modifier ICI, puis lancer generate_ranks_sweep_eval.sh
# ============================================================

# --- Topologie DDP ---
NNODES=6
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.76" # nissan IP
MASTER_PORT=12357            # distinct port to avoid conflicts with training

# Noms des machines (rank 0 en premier)
# Longueur doit être égale à NNODES
RANK_NAMES=("nissan" "1" "2" "3" "4" "5")

# --- Script d'évaluation ---
EVAL_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/evaluate_parallel.py"

# --- Arguments d'évaluation ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
MODEL_SWEEP_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/umx_training/seed=38"
EVAL_BASE_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/evaluations/umx_evaluation/seed=38"
TARGET="vocals"
NITER=0

# --- Flags booléens (mettre à 1 pour activer, 0 pour désactiver) ---
FLAG_IS_WAV=1
FLAG_USE_EDGE=0
