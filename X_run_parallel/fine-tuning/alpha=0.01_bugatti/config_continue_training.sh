#!/usr/bin/env bash
# ============================================================================
#  CONFIG SWEEP DISTRIBUÉ (DDP) POUR CONTINUER LE FINE-TUNING
#  Modifier ce fichier pour configurer les machines et la liste de modèles.
#  Puis lancer generate_ranks_continue_parallel.sh pour générer les lanceurs.
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
# ATTENTION : 'EPOCHS' représente ici le nombre d'époques SUPPLÉMENTAIRES à effectuer.
# Par exemple, si l'entraînement a été interrompu à l'époque 20 et que vous souhaitez
# atteindre un total de 40 époques, mettez EPOCHS=20.
EPOCHS=21
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
# Liste des couples de modèles (BACKBONE_DIR SSM_DIR) dont on poursuit le training.
# Le script déterminera automatiquement le dossier de checkpoints associés dans OUTPUT_BASE.
# Format : "dossier_openunmix dossier_spectrogramme_ssm"
# ─────────────────────────────────────────────────────────────────────────────
COUPLES_CONFIGS=(

    # -30dB
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.5000_l1_4.7027_l2_4.0000 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.5000_l1_4.7027_l2_4.0000"

    # -35dB
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.0500_l1_4.0000_l2_2.9419 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.0500_l1_4.0000_l2_2.9419"

    #-40dB
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.1000_l1_4.0000_l2_1.7152 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.1000_l1_4.0000_l2_1.7152"
    
    #-45dB
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.2_l1_2.1694_l2_1 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.2_l1_2.1694_l2_1"
)
