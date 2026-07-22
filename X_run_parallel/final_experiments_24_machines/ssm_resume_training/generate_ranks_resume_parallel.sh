#!/usr/bin/env bash
# ============================================================================
#  GÉNÉRATEUR DE LANCEURS DDP POUR LA REPRISE DU TRAINING (SSM) SUR 24 MACHINES
#  Usage : bash generate_ranks_resume_parallel.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config_resume_parallel.sh"

# Vérification cohérence
if [ "${#RANK_NAMES[@]}" -ne "$NNODES" ]; then
    echo "ERREUR : RANK_NAMES a ${#RANK_NAMES[@]} éléments mais NNODES=$NNODES"
    exit 1
fi

for RANK in $(seq 0 $((NNODES - 1))); do
    NAME="${RANK_NAMES[$RANK]}"
    # rank 0 → rank0_${NAME}_resume_parallel.sh  |  autres → rank${RANK}_resume_parallel.sh
    if [ "$RANK" -eq 0 ]; then
        OUTFILE="$SCRIPT_DIR/rank0_${NAME}_resume_parallel.sh"
    else
        OUTFILE="$SCRIPT_DIR/rank${RANK}_resume_parallel.sh"
    fi

    # Commencer à écrire le script
    cat > "$OUTFILE" <<HEREDOC
#!/usr/bin/env bash
# Script DDP pour la reprise d'entraînement SSM - Noeud $RANK ($NAME)

set -euo pipefail

# Configurations à reprendre
WINDOW_CONFIGS=(
HEREDOC

    # Écrire les configs dans le script généré
    for cfg in "${WINDOW_CONFIGS[@]}"; do
        echo "    \"$cfg\"" >> "$OUTFILE"
    done

    cat >> "$OUTFILE" <<HEREDOC
)

echo "============================================================"
echo "  Reprise d'entraînement multi-machines — Noeud $RANK ($NAME)"
echo "  Total configurations : \${#WINDOW_CONFIGS[@]}"
echo "============================================================"

for i in "\${!WINDOW_CONFIGS[@]}"; do
    cfg="\${WINDOW_CONFIGS[\$i]}"

    if [ "\$cfg" = "none" ]; then
        RUN_NAME="baseline_no_regul"
        REGUL_ARGS=""
    else
        read -r EPS1 LC1 LC2 <<< "\$cfg"
        RUN_NAME="eps\${EPS1}_l1_\${LC1}_l2_\${LC2}"
        REGUL_ARGS="--regularize_window --epsilon1 \${EPS1} --lambda_coeff_1 \${LC1} --lambda_coeff_2 \${LC2}"
    fi

    OUTPUT_DIR="${OUTPUT_BASE}/\${RUN_NAME}"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [\$((i+1))/\${#WINDOW_CONFIGS[@]}] Lancement : \${RUN_NAME}"
    echo "  Dossier de sortie : \${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "\${OUTPUT_DIR}"

    # Déterminer si on reprend un checkpoint local ou si on démarre d'un checkpoint global
    EXTRA_ARGS=""
    RUN_EPOCHS=${EPOCHS}
    if [ -f "\${OUTPUT_DIR}/${TARGET}.chkpnt" ]; then
        echo "  → Reprise automatique : checkpoint trouvé dans \${OUTPUT_DIR}"
        EXTRA_ARGS="--checkpoint \${OUTPUT_DIR}"
        RUN_EPOCHS=${EPOCHS_RESUME}
    else
        if [ -n "${CHECKPOINT}" ]; then
            EXTRA_ARGS="--checkpoint ${CHECKPOINT}"
            RUN_EPOCHS=${EPOCHS_RESUME}
        fi
    fi

    # Configurer l'environnement pour Conda, ffmpeg, ffprobe, et les sockets NCCL/Gloo
    export PATH="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin:\$PATH"
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=enp,eno,ens,eth,em
    export GLOO_SOCKET_IFNAME=enp,eno,ens,eth,em

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \\
    --nnodes=${NNODES} \\
    --nproc_per_node=${NPROC_PER_NODE} \\
    --node_rank=${RANK} \\
    --master_addr="${MASTER_ADDR}" \\
    --master_port=${MASTER_PORT} \\
    ${TRAIN_SCRIPT} \\
    --root "${ROOT}" \\
    --output "\${OUTPUT_DIR}" \\
    --target "${TARGET}" \\
    --epochs \${RUN_EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --nb-workers ${NB_WORKERS} \\
    --seq-dur ${SEQ_DUR} \\
    --chunk-dur ${CHUNK_DUR} \\
    --nb_magssm_states ${NB_MAGSSM_STATES} \\
    --nfft ${N_FFT} \\
    --nhop ${N_HOP} \\
    --alpha ${ALPHA} \\
    --beta ${BETA} \\
    --eps-stability ${EPS_STABILITY} \\
    --dt-min ${DT_MIN} \\
    --dt-max ${DT_MAX} \\
    --ensure-stability "${ENSURE_STABILITY}" \\
    --re-lower ${RE_LOWER} \\
    --re-upper ${RE_UPPER} \\
    --sigmoid-scale ${SIGMOID_SCALE} \\
    --lr ${LR} \\
    \${EXTRA_ARGS} \\
HEREDOC

    # Ajouter les flags booléens
    [ "$FLAG_IS_WAV" -eq 1 ] && echo "    --is-wav \\" >> "$OUTFILE"
    [ "$FLAG_MEL"    -eq 1 ] && echo "    --mel \\" >> "$OUTFILE"
    [ "$FLAG_AMP"    -eq 1 ] && echo "    --amp \\" >> "$OUTFILE"
    [ "$FLAG_OG"     -eq 1 ] && echo "    --og \\" >> "$OUTFILE"

    # Ajouter le paramètre de régularisation de la fenêtre
    echo "    \${REGUL_ARGS}" >> "$OUTFILE"

    cat >> "$OUTFILE" <<HEREDOC

    echo "  → Reprise de \${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Reprise terminée sur le noeud $RANK !"
echo "============================================================"
HEREDOC

    chmod +x "$OUTFILE"
    echo "✓ Généré : $OUTFILE (node_rank=$RANK)"
done

echo ""
echo "Tous les scripts de reprise SSM ont été générés. Config utilisée :"
echo "  nnodes=$NNODES | master=$MASTER_ADDR:$MASTER_PORT"
echo "  output_base=$OUTPUT_BASE"
