#!/usr/bin/env bash
# ============================================================================
#  GÉNÉRATEUR DE LANCEURS DDP SWEEP POUR LE FINE-TUNING
#  Usage : bash generate_ranks_sweep_parallel.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config_sweep_parallel.sh"

# Vérification cohérence
if [ "${#RANK_NAMES[@]}" -ne "$NNODES" ]; then
    echo "ERREUR : RANK_NAMES a ${#RANK_NAMES[@]} éléments mais NNODES=$NNODES"
    exit 1
fi

for RANK in $(seq 0 $((NNODES - 1))); do
    NAME="${RANK_NAMES[$RANK]}"
    if [ "$RANK" -eq 0 ]; then
        OUTFILE="$SCRIPT_DIR/rank0_${NAME}_sweep_parallel.sh"
    else
        OUTFILE="$SCRIPT_DIR/rank${RANK}_sweep_parallel.sh"
    fi

    # Commencer à écrire le script
    cat > "$OUTFILE" <<HEREDOC
#!/usr/bin/env bash
# Script de sweep DDP Fine-Tuning pour le noeud $RANK ($NAME)

set -euo pipefail

# Couples de modèles à balayer
COUPLES_CONFIGS=(
HEREDOC

    # Écrire la liste des couples dans le script généré
    for couple in "${COUPLES_CONFIGS[@]}"; do
        echo "    \"$couple\"" >> "$OUTFILE"
    done

    cat >> "$OUTFILE" <<HEREDOC
)

echo "============================================================"
echo "  Début du sweep Fine-Tuning DDP — Noeud $RANK ($NAME)"
echo "  Total couples de modèles : \${#COUPLES_CONFIGS[@]}"
echo "============================================================"

for i in "\${!COUPLES_CONFIGS[@]}"; do
    couple="\${COUPLES_CONFIGS[\$i]}"
    
    # Lire le dossier Backbone (OpenUnmix) et le dossier SSM
    read -r BACKBONE_DIR SSM_DIR <<< "\$couple"
    
    # Extraire les noms des répertoires pour construire le nom de sortie
    backbone_name=\$(basename "\$BACKBONE_DIR")
    ssm_name=\$(basename "\$SSM_DIR")
    
    RUN_NAME="backbone_\${backbone_name}_ssm_\${ssm_name}"
    OUTPUT_DIR="${OUTPUT_BASE}/\${RUN_NAME}"

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [\$((i+1))/\${#COUPLES_CONFIGS[@]}] Lancement : \${RUN_NAME}"
    echo "  Backbone : \${BACKBONE_DIR}"
    echo "  SSM      : \${SSM_DIR}"
    echo "  Output   : \${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────"

    mkdir -p "\${OUTPUT_DIR}"

    # Configure environment for conda env, ffmpeg/ffprobe access, and NCCL networking
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
    --model "\${BACKBONE_DIR}" \\
    --ssm-model "\${SSM_DIR}" \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --nb-workers ${NB_WORKERS} \\
    --seq-dur ${SEQ_DUR} \\
    --chunk-dur ${CHUNK_DUR} \\
    --lr ${LR} \\
HEREDOC

    # Option de Freeze
    if [ "$FREEZE_BACKBONE" -eq 1 ]; then
        echo "    --freeze-backbone \\" >> "$OUTFILE"
    else
        echo "    --no-freeze-backbone \\" >> "$OUTFILE"
        echo "    --lr-backbone ${LR_BACKBONE} \\" >> "$OUTFILE"
    fi

    # Option Dataset WAV
    if [ "${FLAG_IS_WAV:-0}" -eq 1 ]; then
        echo "    --is-wav \\" >> "$OUTFILE"
    fi

    # Terminer l'appel de commande
    cat >> "$OUTFILE" <<HEREDOC
    --amp

    echo "  → Configuration \${RUN_NAME} terminée."
done

echo "============================================================"
echo "  Sweep Fine-Tuning DDP terminé sur le noeud $RANK !"
echo "============================================================"
HEREDOC

    chmod +x "$OUTFILE"
    echo "✓ Généré : $OUTFILE (node_rank=$RANK)"
done

echo ""
echo "Tous les scripts de sweep Fine-Tuning sont à jour. Config utilisée :"
echo "  nnodes=$NNODES | master=$MASTER_ADDR:$MASTER_PORT"
echo "  output_base=$OUTPUT_BASE"
