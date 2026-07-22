#!/usr/bin/env bash
# Script de sweep DDP pour le noeud 0 (jaguar)

set -euo pipefail

# Configurations à balayer
WINDOW_CONFIGS=(
    "0.5000   4.7027   4.0000"
)

echo "============================================================"
echo "  Début du sweep multi-machines — Noeud 0 (jaguar)"
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

    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/non_progressive_170bins_alpha4e-3/${RUN_NAME}"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#WINDOW_CONFIGS[@]}] Lancement : ${RUN_NAME}"
    echo "  Dossier de sortie : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    # Déterminer si on reprend un checkpoint local ou si on démarre d'un checkpoint/modèle global
    EXTRA_ARGS=""
    RUN_EPOCHS=60
    if [ -f "${OUTPUT_DIR}/vocals.chkpnt" ]; then
        echo "  → Reprise automatique : checkpoint trouvé dans ${OUTPUT_DIR}"
        EXTRA_ARGS="--checkpoint ${OUTPUT_DIR}"
        RUN_EPOCHS=32
    else
        if [ -n "" ]; then
            EXTRA_ARGS="--checkpoint "
            RUN_EPOCHS=32
        elif [ -n "" ]; then
            EXTRA_ARGS="--model "
        fi
    fi

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \
    --nnodes=6 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --master_addr="129.104.252.72" \
    --master_port=12355 \
    /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_spectrogram_parallel.py \
    --root "/Data/adrien.dubois/musdb18_ds3" \
    --output "${OUTPUT_DIR}" \
    --target "vocals" \
    --epochs ${RUN_EPOCHS} \
    --batch-size 8 \
    --nb-workers 5 \
    --seq-dur 4 \
    --chunk-dur 1 \
    --nb_magssm_states 342 \
    --nfft 340 \
    --nhop 34 \
    --alpha 0.004 \
    --beta 0 \
    --eps-stability 0 \
    --dt-min 0.001 \
    --dt-max 0.1 \
    --lr 0.01 \
    ${EXTRA_ARGS} \
    --is-wav \
    --og \
    --complex_spectrogram \
    --structured_initialisation \
    ${REGUL_ARGS}

    echo "  → Configuration ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Sweep terminé sur le noeud 0 !"
echo "============================================================"
