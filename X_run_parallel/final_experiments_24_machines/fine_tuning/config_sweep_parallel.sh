#!/usr/bin/env bash
# ============================================================================
#  CONFIG SWEEP DISTRIBUÉ (DDP) POUR LE FINE-TUNING SUR 24 MACHINES - MASTER: BUGATTI
#  Ce fichier définit les variables et les configurations pour le fine-tuning.
#  Modifier ce fichier puis lancer generate_ranks_sweep_parallel.sh
# ============================================================================

# --- Topologie DDP (24 machines) ---
NNODES=24
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.65" # IP de bugatti
MASTER_PORT=12356

# Noms des machines (rank 0 en premier)
# bugatti est le master, suivi de 23 machines désignées par leur index/nom
RANK_NAMES=("bugatti" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23")

# --- Script d'entraînement parallélisé ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_magssm_parallel.py"

# --- Arguments Dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/JOINT_OPTIMIZATION_parallel_sweep_-35dB"
TARGET="vocals"

# --- Hyperparamètres d'entraînement ---
EPOCHS=300
BATCH_SIZE=8
NB_WORKERS=2
SEQ_DUR=4
CHUNK_DUR=1

# --- Dataset type ---
FLAG_IS_WAV=1

# --- Scénario de Fine-Tuning ---
# 1 = Scénario A (Freeze backbone, entraîne SSM uniquement)
# 0 = Scénario B (Co-optimisation SSM + backbone avec LRs différenciés)
FREEZE_BACKBONE=0
LR=0.002
LR_BACKBONE=1e-3

# --- Paramètres de stabilité du SSM ---
ENSURE_STABILITY="sigmoid_interval"
RE_LOWER=-0.001  # -1 * 10^(-3)
RE_UPPER=-0.00001  # -1 * 10^(-5)
SIGMOID_SCALE=1.0  # multiplicateur interne de gradient de la sigmoïde (défaut 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DOSSIERS DES MODÈLES UMX ET MAGSSM À OPTIMISER CONJOINTEMENT
# Vous pouvez modifier les deux chemins ci-dessous au besoin :
# ─────────────────────────────────────────────────────────────────────────────

# Modèle UMX (Backbone)
UMX_MODEL_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.5000_l1_4.7027_l2_4.0000"

# Modèle MAGSSM (Spectrogramme)
MAGSSM_MODEL_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/regularized_window_double_exp/NFFT=NSTATES=682_alpha0_TEST_NEW_INITv3/eps0.5000_l1_4.7027_l2_4.0000"

# Liste des couples de modèles (BACKBONE_DIR SSM_DIR)
COUPLES_CONFIGS=(
    "${UMX_MODEL_DIR} ${MAGSSM_MODEL_DIR}"
)
