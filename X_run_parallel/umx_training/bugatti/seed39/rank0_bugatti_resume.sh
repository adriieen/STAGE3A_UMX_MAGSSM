#!/usr/bin/env bash
# Script de reprise DDP pour le noeud 0 (bugatti)

set -euo pipefail

export PATH="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin:$PATH"

# Configuration de reprise
# Format: RUN_NAME | EPOCHS_REMAINING | CHECKPOINT_DIR | REGUL_ARGS
CONFIGS=(
    "eps0.0530_l1_0.4980_l2_0.4365|200||--regularize_window --epsilon1 0.0530 --lambda_coeff_1 0.4980 --lambda_coeff_2 0.4365"
)

echo "============================================================"
echo "  Reprise de l'entraînement DDP — Noeud 0 (bugatti)"
echo "  Total configurations à reprendre : ${#CONFIGS[@]}"
echo "============================================================"

for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r RUN_NAME EPOCHS CHKPNT REGUL_ARGS <<< "${CONFIGS[$i]}"

    OUTPUT_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/umx_training/seed=39/${RUN_NAME}"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#CONFIGS[@]}] Reprise / Lancement : ${RUN_NAME}"
    echo "  Epochs restantes : ${EPOCHS}"
    if [ -n "${CHKPNT}" ]; then
        echo "  Reprise depuis : ${CHKPNT}"
    else
        echo "  Entraînement from scratch (pas de checkpoint)"
    fi
    echo "  Dossier de sortie : ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "${OUTPUT_DIR}"

    CMD=(
        /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun
        --nnodes=6
        --nproc_per_node=1
        --node_rank=0
        --master_addr="129.104.252.65"
        --master_port=12355
        /users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_parallel.py
        --root "/Data/adrien.dubois/musdb18_ds3"
        --output "${OUTPUT_DIR}"
        --target "vocals"
        --epochs "${EPOCHS}"
        --batch-size 16
        --nb-workers 24
        --seq-dur 4
        --nfft 682
        --nhop 34
        --hidden-size 170
        --seed 39
        --is-wav
    )

    if [ -n "${CHKPNT}" ]; then
        CMD+=(--checkpoint "${CHKPNT}")
    fi

    # Ajouter les arguments de régularisation
    if [ -n "${REGUL_ARGS}" ]; then
        # On sépare les arguments proprement
        read -r -a REGUL_ARR <<< "${REGUL_ARGS}"
        CMD+=("${REGUL_ARR[@]}")
    fi

    "${CMD[@]}"

    echo "  → Configuration ${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Reprise terminée sur le noeud 0 !"
echo "============================================================"
