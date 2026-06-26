#!/usr/bin/env bash
# Script de sweep DDP pour le noeud 2 (2)

set -euo pipefail

# Configurations à balayer
WINDOW_CONFIGS=(
    "none"
    "0.053    0.5      0.257"
    "0.1055   0.5918   0.4621"
    "0.2      0.9592   1"
    "0.1265   0.1796   0.0778"
)

echo "============================================================"
echo "  Début du sweep multi-machines — Noeud 2 (2)"
echo "  Total configurations : ${#WINDOW_CONFIGS[@]}"
echo "============================================================"

for i in "${!WINDOW_CONFIGS[@]}"; do
    cfg="${WINDOW_CONFIGS[$i]}"

    if [ "$cfg" = "none" ]; then
        RUN_NAME="baseline_no_regul"
        REGUL_ARGS=""
    else
        read -r EPS1 LC1 LC2 <<< "$cfg"
        RUN_NAME="eps${EPS1}_l1_${LC1}_l2_${LC2}"
        REGUL_ARGS="--regularize_window --epsilon1 ${EPS1} --lambda_coeff_1 ${LC1} --lambda_coeff_2 ${LC2}"
    fi

    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/n_states=1nbins/512bins_alpha5e0_sweep/${RUN_NAME}"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#WINDOW_CONFIGS[@]}] Lancement : ${RUN_NAME}"
    echo "  Dossier de sortie : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    torchrun \
    --nnodes=6 \
    --nproc_per_node=1 \
    --node_rank=2 \
    --master_addr="129.104.252.76" \
    --master_port=12355 \
    /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_spectrogram_parallel.py \
    --root "/Data/adrien.dubois/musdb18_ds3" \
    --output "${OUTPUT_DIR}" \
    --target "vocals" \
    --epochs 150 \
    --batch-size 8 \
    --nb-workers 5 \
    --seq-dur 4 \
    --chunk-dur 1 \
    --nb_magssm_states 682 \
    --nfft 682 \
    --nhop 102 \
    --alpha 5 \
    --beta 1 \
    --eps-stability 0 \
    --dt-min 0.001 \
    --dt-max 0.1 \
    --is-wav \
    --mel \
    ${REGUL_ARGS}

    echo "  → Configuration ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Sweep terminé sur le noeud 2 !"
echo "============================================================"
