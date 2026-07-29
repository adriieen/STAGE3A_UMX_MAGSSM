#!/usr/bin/env bash
# Script de fine-tuning DDP pour le noeud 10 (10)

set -euo pipefail

WINDOW_CONFIGS=(
    "0.5000   4.7027   4.0000"
)

echo "============================================================"
echo "  Début du Fine-Tuning Multi-Machines — Noeud 10 (10)"
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

    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine_tuning_july/vocals_pipeline/joint_optimization/${RUN_NAME}"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#WINDOW_CONFIGS[@]}] Lancement fine-tuning : ${RUN_NAME}"
    echo "  Dossier de sortie : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    # Construction des arguments de reprise ou de modèles pré-entraînés
    EXTRA_ARGS=""
    RUN_EPOCHS=400

    if [ -f "${OUTPUT_DIR}/vocals.chkpnt" ]; then
        echo "  → Reprise automatique : checkpoint combiné trouvé dans ${OUTPUT_DIR}"
        EXTRA_ARGS="--checkpoint ${OUTPUT_DIR}"
        RUN_EPOCHS=40
    else
        if [ -n "" ]; then
            EXTRA_ARGS="--checkpoint "
            RUN_EPOCHS=40
        else
            if [ -n "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/[small_lr]non_progressive_342bins_alpha0_structured_init/eps0.5000_l1_4.7027_l2_4.0000" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --ssm-model /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/[small_lr]non_progressive_342bins_alpha0_structured_init/eps0.5000_l1_4.7027_l2_4.0000"
            fi
            if [ -n "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_decoders/[341bins_modified_nhop_low_lr]spectrogram_loss_modified_ssm/eps0.5000_l1_4.7027_l2_4.0000" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --decoder-model /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_decoders/[341bins_modified_nhop_low_lr]spectrogram_loss_modified_ssm/eps0.5000_l1_4.7027_l2_4.0000"
            fi
            if [ -n "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/post_soutenance/umx_training/[28-07]seed=42/eps0.5000_l1_4.7027_l2_4.0000" ]; then
                EXTRA_ARGS="${EXTRA_ARGS} --model /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/post_soutenance/umx_training/[28-07]seed=42/eps0.5000_l1_4.7027_l2_4.0000"
            fi
        fi
    fi

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \
    --nnodes=18 \
    --nproc_per_node=1 \
    --node_rank=10 \
    --master_addr="129.104.252.65" \
    --master_port=12355 \
    /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_magssm_parallel.py \
    --root "/Data/adrien.dubois/musdb18_ds3" \
    --output "${OUTPUT_DIR}" \
    --target "vocals" \
    --epochs ${RUN_EPOCHS} \
    --batch-size 6 \
    --nb-workers 5 \
    --seq-dur 4 \
    --chunk-dur 1 \
    --alpha 0 \
    --beta 0 \
    --eps-stability 0 \
    --dt-min 0.001 \
    --dt-max 0.1 \
    --lr 0.0005 \
    --lr-backbone 0.001 \
    --is-wav \
    ${EXTRA_ARGS} \
    --no-freeze-backbone \
    --fft_kernel \
    --og \
    --complex_spectrogram \
    --structured_initialisation \
    ${REGUL_ARGS}

    echo "  → Configuration ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Fine-tuning terminé sur le noeud 10 !"
echo "============================================================"
