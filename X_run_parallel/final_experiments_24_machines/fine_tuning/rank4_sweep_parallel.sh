#!/usr/bin/env bash
# Script de sweep DDP Fine-Tuning pour le noeud 4 (4)

set -euo pipefail

# Couples de modèles à balayer
COUPLES_CONFIGS=(
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.5000_l1_4.7027_l2_4.0000 /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/regularized_window_double_exp/NFFT=NSTATES=682_alpha0_TEST_NEW_INITv3/eps0.5000_l1_4.7027_l2_4.0000"
)

echo "============================================================"
echo "  Début du sweep Fine-Tuning DDP — Noeud 4 (4)"
echo "  Total couples de modèles : ${#COUPLES_CONFIGS[@]}"
echo "============================================================"

for i in "${!COUPLES_CONFIGS[@]}"; do
    couple="${COUPLES_CONFIGS[$i]}"
    
    # Lire le dossier Backbone (OpenUnmix) et le dossier SSM
    read -r BACKBONE_DIR SSM_DIR <<< "$couple"
    
    # Extraire les noms des répertoires pour construire le nom de sortie
    backbone_name=$(basename "$BACKBONE_DIR")
    ssm_name=$(basename "$SSM_DIR")
    
    RUN_NAME="backbone_${backbone_name}_ssm_${ssm_name}"
    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/JOINT_OPTIMIZATION_parallel_sweep_-35dB/${RUN_NAME}"

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#COUPLES_CONFIGS[@]}] Lancement : ${RUN_NAME}"
    echo "  Backbone : ${BACKBONE_DIR}"
    echo "  SSM      : ${SSM_DIR}"
    echo "  Output   : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    # Configure environment for conda env, ffmpeg/ffprobe access, and NCCL networking
    export PATH="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin:$PATH"
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=enp,eno,ens,eth,em
    export GLOO_SOCKET_IFNAME=enp,eno,ens,eth,em

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \
    --nnodes=24 \
    --nproc_per_node=1 \
    --node_rank=4 \
    --master_addr="129.104.252.65" \
    --master_port=12356 \
    /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_magssm_parallel.py \
    --root "/Data/adrien.dubois/musdb18_ds3" \
    --output "${OUTPUT_DIR}" \
    --target "vocals" \
    --model "${BACKBONE_DIR}" \
    --ssm-model "${SSM_DIR}" \
    --epochs 300 \
    --batch-size 8 \
    --nb-workers 2 \
    --seq-dur 4 \
    --chunk-dur 1 \
    --ensure-stability "sigmoid_interval" \
    --re-lower -0.001 \
    --re-upper -0.00001 \
    --sigmoid-scale 1.0 \
    --lr 0.002 \
    --no-freeze-backbone \
    --lr-backbone 1e-3 \
    --is-wav \
    --amp

    echo "  → Configuration ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Sweep Fine-Tuning DDP terminé sur le noeud 4 !"
echo "============================================================"
