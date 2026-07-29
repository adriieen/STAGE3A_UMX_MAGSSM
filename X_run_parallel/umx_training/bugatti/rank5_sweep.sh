#!/usr/bin/env bash
# Script de sweep DDP pour le noeud 5 (5)

set -euo pipefail

export PATH="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin:$PATH"

# Configurations à balayer
WINDOW_CONFIGS=(
    "0.1055   0.1000   0.1291"
)

echo "============================================================"
echo "  Début du sweep multi-machines — Noeud 5 (5)"
echo "  Total configurations : ${#WINDOW_CONFIGS[@]}"
echo "============================================================"

for i in "${!WINDOW_CONFIGS[@]}"; do
    cfg="${WINDOW_CONFIGS[$i]}"

    if [ "$cfg" = "none" ] || [ "$cfg" = "0 0 0" ]; then
        if [ "$cfg" = "0 0 0" ]; then
            read -r EPS1 LC1 LC2 <<< "$cfg"
            RUN_NAME="baseline_no_regul"
            REGUL_ARGS="--regularize_window --epsilon1 ${EPS1} --lambda_coeff_1 ${LC1} --lambda_coeff_2 ${LC2}"
        else
            RUN_NAME="baseline_no_regul"
            REGUL_ARGS=""
        fi
    else
        read -r EPS1 LC1 LC2 <<< "$cfg"
        RUN_NAME="eps${EPS1}_l1_${LC1}_l2_${LC2}"
        REGUL_ARGS="--regularize_window --epsilon1 ${EPS1} --lambda_coeff_1 ${LC1} --lambda_coeff_2 ${LC2}"
    fi

    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/post_soutenance/umx_training/[28-07]seed=42/${RUN_NAME}"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#WINDOW_CONFIGS[@]}] Lancement : ${RUN_NAME}"
    echo "  Dossier de sortie : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \
    --nnodes=6 \
    --nproc_per_node=1 \
    --node_rank=5 \
    --master_addr="129.104.252.65" \
    --master_port=12355 \
    /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_parallel.py \
    --root "/Data/adrien.dubois/musdb18_ds3" \
    --output "${OUTPUT_DIR}" \
    --target "vocals" \
    --epochs 400 \
    --batch-size 16 \
    --nb-workers 24 \
    --seq-dur 4 \
    --nfft 680 \
    --nhop 68 \
    --hidden-size 170 \
    --seed 42 \
    --lr 0.002 \
    --is-wav \
    ${REGUL_ARGS}

    echo "  → Configuration ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Sweep terminé sur le noeud 5 !"
echo "============================================================"
