#!/usr/bin/env bash
# ============================================================
#  GÉNÉRATEUR — recrée tous les rankN.sh à partir de config.sh
#  Usage : bash generate_ranks.sh
#  (à lancer depuis le dossier X_run_parallel/ ou depuis la racine)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Vérification cohérence
if [ "${#RANK_NAMES[@]}" -ne "$NNODES" ]; then
    echo "ERREUR : RANK_NAMES a ${#RANK_NAMES[@]} éléments mais NNODES=$NNODES"
    exit 1
fi

# Construction des flags booléens
FLAGS=""
[ "$FLAG_IS_WAV"        -eq 1 ] && FLAGS+=$'\\\n--is-wav \\'
[ "$FLAG_MEL"           -eq 1 ] && FLAGS+=$'\\\n--mel \\'
[ "$FLAG_AMP"           -eq 1 ] && FLAGS+=$'\\\n--amp \\'
[ "$FLAG_OG"            -eq 1 ] && FLAGS+=$'\\\n--og \\'

for RANK in $(seq 0 $((NNODES - 1))); do
    NAME="${RANK_NAMES[$RANK]}"
    # rank 0 → rank0_nissan.sh  |  autres → rank1.sh, rank2.sh, …
    if [ "$RANK" -eq 0 ]; then
        OUTFILE="$SCRIPT_DIR/rank0_${NAME}.sh"
    else
        OUTFILE="$SCRIPT_DIR/rank${RANK}.sh"
    fi

    cat > "$OUTFILE" <<HEREDOC
torchrun \\
--nnodes=${NNODES} \\
--nproc_per_node=${NPROC_PER_NODE} \\
--node_rank=${RANK} \\
--master_addr="${MASTER_ADDR}" \\
--master_port=${MASTER_PORT} \\
${TRAIN_SCRIPT} \\
--root ${ROOT} \\
--output ${OUTPUT} \\
--target ${TARGET} \\
--epochs ${EPOCHS} \\
--batch-size ${BATCH_SIZE} \\
--nb-workers ${NB_WORKERS} \\
--seq-dur ${SEQ_DUR} \\
--chunk-dur ${CHUNK_DUR} \\
--nb_magssm_states ${NB_MAGSSM_STATES} \\
--nfft ${N_FFT} \\
--nhop ${N_HOP} \\
--ds ${DOWNSAMPLING} \\
--alpha ${ALPHA} \\
--beta ${BETA} \\
--eps-stability ${EPS_STABILITY} \\
--dt-min ${DT_MIN} \\
--dt-max ${DT_MAX} \\
HEREDOC

    # Ajouter les arguments optionnels (fine-tuning / reprise)
    [ -n "$MODEL" ]      && echo "--model ${MODEL} \\" >> "$OUTFILE"
    [ -n "$CHECKPOINT" ] && echo "--checkpoint ${CHECKPOINT} \\" >> "$OUTFILE"

    # Ajouter les flags booléens
    [ "$FLAG_IS_WAV"         -eq 1 ] && echo "--is-wav \\" >> "$OUTFILE"
    [ "$FLAG_MEL"            -eq 1 ] && echo "--mel \\" >> "$OUTFILE"
    [ "$FLAG_AMP"            -eq 1 ] && echo "--amp \\" >> "$OUTFILE"
    [ "$FLAG_OG"             -eq 1 ] && echo "--og \\" >> "$OUTFILE"

    chmod +x "$OUTFILE"
    echo "✓ Généré : $OUTFILE (node_rank=$RANK)"
done

echo ""
echo "Tous les scripts sont à jour. Config utilisée :"
echo "  nnodes=$NNODES | master=$MASTER_ADDR:$MASTER_PORT"
echo "  output=$OUTPUT"
