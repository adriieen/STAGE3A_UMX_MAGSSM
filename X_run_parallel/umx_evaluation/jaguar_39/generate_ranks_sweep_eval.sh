#!/usr/bin/env bash
# ============================================================
#  GÉNÉRATEUR SWEEP EVAL — recrée tous les rankN_sweep_eval.sh
#  Usage : bash generate_ranks_sweep_eval.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config_sweep_eval.sh"

# Vérification cohérence
if [ "${#RANK_NAMES[@]}" -ne "$NNODES" ]; then
    echo "ERREUR : RANK_NAMES a ${#RANK_NAMES[@]} éléments mais NNODES=$NNODES"
    exit 1
fi

for RANK in $(seq 0 $((NNODES - 1))); do
    NAME="${RANK_NAMES[$RANK]}"
    # rank 0 → rank0_${NAME}_sweep_eval.sh  |  autres → rank${RANK}_sweep_eval.sh
    if [ "$RANK" -eq 0 ]; then
        OUTFILE="$SCRIPT_DIR/rank0_${NAME}_sweep_eval.sh"
    else
        OUTFILE="$SCRIPT_DIR/rank${RANK}_sweep_eval.sh"
    fi

    # Commencer à écrire le script
    cat > "$OUTFILE" <<HEREDOC
#!/usr/bin/env bash
# Script de sweep d'évaluation DDP pour le noeud $RANK ($NAME)

set -euo pipefail

export PATH="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin:\$PATH"

# Répertoire de sweep de modèles
MODEL_SWEEP_DIR="${MODEL_SWEEP_DIR}"
EVAL_BASE_DIR="${EVAL_BASE_DIR}"

echo "============================================================"
echo "  Début du sweep d'évaluation DDP — Noeud $RANK ($NAME)"
echo "  Dossier des modèles : \${MODEL_SWEEP_DIR}"
echo "============================================================"

# Récupérer et trier la liste des modèles pour être sûr que tous les noeuds
# parcourent les mêmes configurations dans le même ordre.
model_dirs=()
for model_dir in "\${MODEL_SWEEP_DIR}"/*; do
    if [ -d "\${model_dir}" ] && [ -f "\${model_dir}/separator.json" ]; then
        model_dirs+=("\${model_dir}")
    fi
done

IFS=\$'\\n' sorted_model_dirs=(\$(sort <<<"\${model_dirs[*]}"))
unset IFS

echo "Total modèles à évaluer : \${#sorted_model_dirs[@]}"

for i in "\${!sorted_model_dirs[@]}"; do
    model_dir="\${sorted_model_dirs[\$i]}"
    model_name=\$(basename "\${model_dir}")
    evaldir="\${EVAL_BASE_DIR}/\${model_name}"

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [\$((i+1))/\${#sorted_model_dirs[@]}] Évaluation de : \${model_name}"
    echo "  Dossier modèle : \${model_dir}"
    echo "  Dossier eval   : \${evaldir}"
    echo "────────────────────────────────────────────────────"

    # S'assurer que le dossier d'évaluation existe
    mkdir -p "\${evaldir}"

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \\
    --nnodes=${NNODES} \\
    --nproc_per_node=${NPROC_PER_NODE} \\
    --node_rank=${RANK} \\
    --master_addr="${MASTER_ADDR}" \\
    --master_port=${MASTER_PORT} \\
    "${EVAL_SCRIPT}" \\
    --model "\${model_dir}" \\
    --root "${ROOT}" \\
    --target "${TARGET}" \\
    --evaldir "\${evaldir}" \\
    --niter ${NITER} \\
HEREDOC

    # Ajouter les flags optionnels
    [ "${FLAG_IS_WAV}" -eq 1 ] && echo "    --is-wav \\" >> "$OUTFILE"
    [ "${FLAG_USE_EDGE}" -eq 1 ] && echo "    --use_edge \\" >> "$OUTFILE"

    # Terminer la commande torchrun
    cat >> "$OUTFILE" <<HEREDOC
    --cores 1

    echo "  → Évaluation de \${model_name} terminée."
done

echo "============================================================"
echo "  Sweep d'évaluation terminé sur le noeud $RANK !"
echo "============================================================"
HEREDOC

    chmod +x "$OUTFILE"
    echo "✓ Généré : $OUTFILE (node_rank=$RANK)"
done

echo ""
echo "Tous les scripts de sweep d'évaluation sont à jour. Config utilisée :"
echo "  nnodes=$NNODES | master=$MASTER_ADDR:$MASTER_PORT"
echo "  model_sweep_dir=$MODEL_SWEEP_DIR"
echo "  eval_base_dir=$EVAL_BASE_DIR"
