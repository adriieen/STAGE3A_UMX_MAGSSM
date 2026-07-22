#!/usr/bin/env bash
# Script de sweep DDP Fine-Tuning pour le noeud 5 (5)

set -euo pipefail

# Couples de modèles à balayer
COUPLES_CONFIGS=(
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/bilstm_classic /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/alpha=0.1/baseline_no_regul"
)

echo "============================================================"
echo "  Début du sweep Fine-Tuning DDP — Noeud 5 (5)"
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
    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine-tuning/alpha=0.1/separator_frozen/${RUN_NAME}"

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
    --nnodes=6 \
    --nproc_per_node=1 \
    --node_rank=5 \
    --master_addr="129.104.252.76" \
    --master_port=12356 \
    /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_magssm_parallel.py \
    --root "/Data/adrien.dubois/musdb18_ds3" \
    --output "${OUTPUT_DIR}" \
    --target "vocals" \
    --model "${BACKBONE_DIR}" \
    --ssm-model "${SSM_DIR}" \
    --epochs 40 \
    --batch-size 8 \
    --nb-workers 2 \
    --seq-dur 4 \
    --chunk-dur 1 \
    --lr 0.001 \
    --freeze-backbone \
    --is-wav \
    --amp

    echo "  → Configuration ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Sweep Fine-Tuning DDP terminé sur le noeud 5 !"
echo "============================================================"
