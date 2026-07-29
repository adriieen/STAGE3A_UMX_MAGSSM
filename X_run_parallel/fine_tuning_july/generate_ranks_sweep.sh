#!/usr/bin/env bash
# ============================================================
#  GÉNÉRATEUR SWEEP FINE-TUNING
#  Génère tous les scripts rankN_sweep.sh à partir de config_sweep.sh
#  Usage : bash generate_ranks_sweep.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config_sweep.sh"

# Control checks
if [ "${#RANK_NAMES[@]}" -ne "$NNODES" ]; then
    echo "ERREUR : RANK_NAMES a ${#RANK_NAMES[@]} éléments mais NNODES=$NNODES"
    exit 1
fi

for RANK in $(seq 0 $((NNODES - 1))); do
    NAME="${RANK_NAMES[$RANK]}"
    if [ "$RANK" -eq 0 ]; then
        OUTFILE="$SCRIPT_DIR/rank0_${NAME}_sweep.sh"
    else
        OUTFILE="$SCRIPT_DIR/rank${RANK}_sweep.sh"
    fi

    cat > "$OUTFILE" <<HEREDOC
#!/usr/bin/env bash
# Script de fine-tuning DDP pour le noeud $RANK ($NAME)

set -euo pipefail

WINDOW_CONFIGS=(
HEREDOC

    for cfg in "${WINDOW_CONFIGS[@]}"; do
        echo "    \"$cfg\"" >> "$OUTFILE"
    done

    cat >> "$OUTFILE" <<HEREDOC
)

echo "============================================================"
echo "  Début du Fine-Tuning Multi-Machines — Noeud $RANK ($NAME)"
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
    echo "  [\$((i+1))/\${#WINDOW_CONFIGS[@]}] Lancement fine-tuning : \${RUN_NAME}"
    echo "  Dossier de sortie : \${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "\${OUTPUT_DIR}"

    # Construction des arguments de reprise ou de modèles pré-entraînés
    EXTRA_ARGS=""
    RUN_EPOCHS=${EPOCHS}

    if [ -f "\${OUTPUT_DIR}/${TARGET}.chkpnt" ]; then
        echo "  → Reprise automatique : checkpoint combiné trouvé dans \${OUTPUT_DIR}"
        EXTRA_ARGS="--checkpoint \${OUTPUT_DIR}"
        RUN_EPOCHS=${EPOCHS_RESUME}
    else
        if [ -n "${CHECKPOINT}" ]; then
            EXTRA_ARGS="--checkpoint ${CHECKPOINT}"
            RUN_EPOCHS=${EPOCHS_RESUME}
        else
            if [ -n "${SPECTROGRAM_MODEL}" ]; then
                EXTRA_ARGS="\${EXTRA_ARGS} --ssm-model ${SPECTROGRAM_MODEL}"
            fi
            if [ -n "${DECODER_MODEL}" ]; then
                EXTRA_ARGS="\${EXTRA_ARGS} --decoder-model ${DECODER_MODEL}"
            fi
            if [ -n "${UMX_MODEL}" ]; then
                EXTRA_ARGS="\${EXTRA_ARGS} --model ${UMX_MODEL}"
            fi
        fi
    fi

    ${TORCHRUN_BIN} \\
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
    --alpha ${ALPHA} \\
    --beta ${BETA} \\
    --eps-stability ${EPS_STABILITY} \\
    --dt-min ${DT_MIN} \\
    --dt-max ${DT_MAX} \\
    --lr ${LEARNING_RATE} \\
    --lr-backbone ${LR_BACKBONE} \\
    --is-wav \\
    \${EXTRA_ARGS} \\
HEREDOC

    # Add optional architecture overrides if specified in config
    [ -n "${NB_MAGSSM_STATES:-}" ] && echo "    --nb_magssm_states ${NB_MAGSSM_STATES} \\" >> "$OUTFILE"
    [ -n "${N_FFT:-}" ]            && echo "    --nfft ${N_FFT} \\" >> "$OUTFILE"
    [ -n "${N_HOP:-}" ]            && echo "    --nhop ${N_HOP} \\" >> "$OUTFILE"

    # Adding boolean flags
    [ "${FLAG_FREEZE_BACKBONE:-1}" -eq 1 ] && echo "    --freeze-backbone \\" >> "$OUTFILE" || echo "    --no-freeze-backbone \\" >> "$OUTFILE"
    [ "${FLAG_FREEZE_SSM:-0}"      -eq 1 ] && echo "    --freeze-ssm \\" >> "$OUTFILE"
    [ "${FLAG_FREEZE_DECODER:-0}"  -eq 1 ] && echo "    --freeze-decoder \\" >> "$OUTFILE"
    [ "${FLAG_PROGRESSIVE:-0}"     -eq 1 ] && echo "    --progressive \\" >> "$OUTFILE"
    [ "${FLAG_FFT_KERNEL:-1}"      -eq 1 ] && echo "    --fft_kernel \\" >> "$OUTFILE"
    [ "${FLAG_MEL:-0}"             -eq 1 ] && echo "    --mel \\" >> "$OUTFILE"
    [ "${FLAG_AMP:-0}"             -eq 1 ] && echo "    --amp \\" >> "$OUTFILE"
    [ "${FLAG_OG:-1}"              -eq 1 ] && echo "    --og \\" >> "$OUTFILE"
    [ "${FLAG_COMPLEX_SPECTROGRAM:-1}" -eq 1 ] && echo "    --complex_spectrogram \\" >> "$OUTFILE"
    [ "${FLAG_STRUCTURED_INITIALISATION:-1}" -eq 1 ] && echo "    --structured_initialisation \\" >> "$OUTFILE"
    [ "${FLAG_PHASE_CORRECTION:-0}" -eq 1 ] && echo "    --phase-correction \\" >> "$OUTFILE"

    echo "    \${REGUL_ARGS}" >> "$OUTFILE"

    cat >> "$OUTFILE" <<HEREDOC

    echo "  → Configuration \${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Fine-tuning terminé sur le noeud $RANK !"
echo "============================================================"
HEREDOC

    chmod +x "$OUTFILE"
    echo "✓ Généré : $OUTFILE (node_rank=$RANK)"
done

echo ""
echo "Tous les scripts de fine-tuning sont à jour dans :"
echo "  $SCRIPT_DIR"
echo "Config utilisée :"
echo "  nnodes=$NNODES | master=$MASTER_ADDR:$MASTER_PORT"
echo "  output_base=$OUTPUT_BASE"
echo "  SSM Encoder model : ${SPECTROGRAM_MODEL:-'(aucun - rand init)'}"
echo "  SSM Decoder model : ${DECODER_MODEL:-'(aucun - rand init)'}"
echo "  OpenUnmix backbone: ${UMX_MODEL:-'(aucun - rand init)'}"
