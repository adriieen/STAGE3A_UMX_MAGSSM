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

# --- Script d'entraînement parallélisé ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_magssm_parallel.py"

# --- Arguments Dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/JOINT_OPTIMIZATION_parallel_sweep"
TARGET="vocals"

# --- Hyperparamètres d'entraînement ---
EPOCHS=40
BATCH_SIZE=8
NB_WORKERS=2
SEQ_DUR=4
CHUNK_DUR=1

# --- Dataset type ---
FLAG_IS_WAV=1

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

    # ----------------------------------- SCENARIO A --------------------------------------

    # # -30dB
    # "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.5000_l1_4.7027_l2_4.0000 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.5000_l1_4.7027_l2_4.0000"


    # # -35dB
    # "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.0500_l1_4.0000_l2_2.9419 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.0500_l1_4.0000_l2_2.9419"

    # #-40dB
    # "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.1000_l1_4.0000_l2_1.7152 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.1000_l1_4.0000_l2_1.7152"
    
    # #-45dB
    # "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.2_l1_2.1694_l2_1 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.2_l1_2.1694_l2_1"
    # # Ajoutez d'autres lignes ici au besoin


    # -------------------------------------- SCENARIO B --------------------------------------

    # -30dB

    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.5000_l1_4.7027_l2_4.0000 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/parallel_sweep/backbone_eps0.5000_l1_4.7027_l2_4.0000_ssm_eps0.5000_l1_4.7027_l2_4.0000"


    # -35 dB
    
        "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.0500_l1_4.0000_l2_2.9419 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/parallel_sweep/backbone_eps0.0500_l1_4.0000_l2_2.9419_ssm_eps0.0500_l1_4.0000_l2_2.9419"
    
    # -40 dB
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.1000_l1_4.0000_l2_1.7152 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/parallel_sweep/backbone_eps0.1000_l1_4.0000_l2_1.7152_ssm_eps0.1000_l1_4.0000_l2_1.7152"
    
    # -45 dB
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.2_l1_2.1694_l2_1 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/parallel_sweep/backbone_eps0.2_l1_2.1694_l2_1_ssm_eps0.2_l1_2.1694_l2_1"

)
