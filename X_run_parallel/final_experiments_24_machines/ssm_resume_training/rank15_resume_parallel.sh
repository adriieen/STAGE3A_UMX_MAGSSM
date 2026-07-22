#!/usr/bin/env bash
# Script DDP pour la reprise d'entraînement SSM - Noeud 15 (15)

set -euo pipefail

# Configurations à reprendre
WINDOW_CONFIGS=(
    "0 0 0"
)

echo "============================================================"
echo "  Reprise d'entraînement multi-machines — Noeud 15 (15)"
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

    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/regularized_window_double_exp/NSTATES=684_test_14_07_V2/${RUN_NAME}"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#WINDOW_CONFIGS[@]}] Lancement : ${RUN_NAME}"
    echo "  Dossier de sortie : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    # Déterminer si on reprend un checkpoint local ou si on démarre d'un checkpoint global
    EXTRA_ARGS=""
    RUN_EPOCHS=300
    if [ -f "${OUTPUT_DIR}/vocals.chkpnt" ]; then
        echo "  → Reprise automatique : checkpoint trouvé dans ${OUTPUT_DIR}"
        EXTRA_ARGS="--checkpoint ${OUTPUT_DIR}"
        RUN_EPOCHS=150
    else
        if [ -n "" ]; then
            EXTRA_ARGS="--checkpoint "
            RUN_EPOCHS=150
        fi
    fi

    # Configurer l'environnement pour Conda, ffmpeg, ffprobe, et les sockets NCCL/Gloo
    export PATH="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin:$PATH"
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=enp,eno,ens,eth,em
    export GLOO_SOCKET_IFNAME=enp,eno,ens,eth,em

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \
    --nnodes=24 \
    --nproc_per_node=1 \
    --node_rank=15 \
    --master_addr="129.104.252.65" \
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
    --nb_magssm_states 684 \
    --nfft 682 \
    --nhop 34 \
    --alpha 0 \
    --beta 0 \
    --eps-stability 0 \
    --dt-min 0.001 \
    --dt-max 0.1 \
    --ensure-stability "relu" \
    --re-lower -0.0005 \
    --re-upper -0.0000005 \
    --sigmoid-scale 1.0 \
    --lr 0.004 \
    ${EXTRA_ARGS} \
    --is-wav \
    --og \
    ${REGUL_ARGS}

    echo "  → Reprise de ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Reprise terminée sur le noeud 15 !"
echo "============================================================"
