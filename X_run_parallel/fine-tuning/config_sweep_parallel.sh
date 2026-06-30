#!/usr/bin/env bash
# ============================================================================
#  CONFIG SWEEP DISTRIBUÉ (DDP) POUR LE FINE-TUNING
#  Modifier ce fichier pour configurer les machines et la liste de modèles.
#  Puis lancer generate_ranks_sweep_parallel.sh pour régénérer les lanceurs.
# ============================================================================

# --- Topologie DDP (6 machines) ---
NNODES=6
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.76"
MASTER_PORT=12356

# Noms des machines (rank 0 en premier)
RANK_NAMES=("nissan" "1" "2" "3" "4" "5")

# --- Script d'entraînement parallélisé ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_magssm_parallel.py"

# --- Arguments Dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/parallel_sweep"
TARGET="vocals"

# --- Hyperparamètres d'entraînement ---
EPOCHS=1000
BATCH_SIZE=20
NB_WORKERS=10
SEQ_DUR=6.0
CHUNK_DUR=6.0

# --- Scénario de Fine-Tuning ---
# 1 = Scénario A (Freeze backbone, entraîne SSM uniquement)
# 0 = Scénario B (Co-optimisation SSM + backbone avec LRs différenciés)
FREEZE_BACKBONE=1
LR=0.001
LR_BACKBONE=1e-5

# ─────────────────────────────────────────────────────────────────────────────
# Liste des couples de modèles (BACKBONE_DIR SSM_DIR)
# Format : "dossier_openunmix dossier_spectrogramme_ssm"
# ─────────────────────────────────────────────────────────────────────────────
COUPLES_CONFIGS=(
    "/home/adubois/openunmix/OpenUnmix/outputs/regul_window_sweep/-20-->-40/eps0.1000_l1_4.0000_l2_1.7152 /home/adubois/openunmix/OpenUnmix/outputs/trainable_spectograms/14.7kHz/regularized_window_double_exp/NFFT=NSTATES=682/eps0.1000_l1_4.0000_l2_1.7152"
    # Ajoutez d'autres lignes ici au besoin
)
