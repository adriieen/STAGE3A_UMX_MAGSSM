#!/usr/bin/env bash
# Script de sweep d'évaluation DDP pour le noeud 1 (1)

set -euo pipefail

export PATH="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin:$PATH"

# Répertoire de sweep de modèles
MODEL_SWEEP_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/umx_training/seed=38"
EVAL_BASE_DIR="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/evaluations/umx_evaluation/seed=38"

echo "============================================================"
echo "  Début du sweep d'évaluation DDP — Noeud 1 (1)"
echo "  Dossier des modèles : ${MODEL_SWEEP_DIR}"
echo "============================================================"

# Récupérer et trier la liste des modèles pour être sûr que tous les noeuds
# parcourent les mêmes configurations dans le même ordre.
model_dirs=()
for model_dir in "${MODEL_SWEEP_DIR}"/*; do
    if [ -d "${model_dir}" ] && [ -f "${model_dir}/separator.json" ]; then
        model_dirs+=("${model_dir}")
    fi
done

IFS=$'\n' sorted_model_dirs=($(sort <<<"${model_dirs[*]}"))
unset IFS

echo "Total modèles à évaluer : ${#sorted_model_dirs[@]}"

for i in "${!sorted_model_dirs[@]}"; do
    model_dir="${sorted_model_dirs[$i]}"
    model_name=$(basename "${model_dir}")
    evaldir="${EVAL_BASE_DIR}/${model_name}"

    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  [$((i+1))/${#sorted_model_dirs[@]}] Évaluation de : ${model_name}"
    echo "  Dossier modèle : ${model_dir}"
    echo "  Dossier eval   : ${evaldir}"
    echo "────────────────────────────────────────────────────"

    # S'assurer que le dossier d'évaluation existe
    mkdir -p "${evaldir}"

    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun \
    --nnodes=6 \
    --nproc_per_node=1 \
    --node_rank=1 \
    --master_addr="129.104.252.76" \
    --master_port=12357 \
    "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/evaluate_parallel.py" \
    --model "${model_dir}" \
    --root "/Data/adrien.dubois/musdb18_ds3" \
    --target "vocals" \
    --evaldir "${evaldir}" \
    --niter 0 \
    --is-wav \
    --cores 1

    echo "  → Évaluation de ${model_name} terminée."
done

echo "============================================================"
echo "  Sweep d'évaluation terminé sur le noeud 1 !"
echo "============================================================"
