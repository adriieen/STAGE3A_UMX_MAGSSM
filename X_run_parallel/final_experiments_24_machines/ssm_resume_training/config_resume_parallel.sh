#!/usr/bin/env bash
# ============================================================================
#  CONFIG REPRISE DE TRAINING (SSM) SUR 24 MACHINES - MASTER: BUGATTI
#  Ce fichier définit les variables et les configurations pour la reprise.
#  Modifier ce fichier puis lancer generate_ranks_resume_parallel.sh
# ============================================================================

# --- Reprise / Checkpoint ---
# Le dossier contenant vocals.chkpnt et vocals.json du modèle trainable spectrograms
# CHECKPOINT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/regularized_window_double_exp/NFFT=NSTATES=682_alpha0.01/eps0.0500_l1_4.0000_l2_2.9419"
CHECKPOINT=""

# --- Topologie DDP (24 machines) ---
NNODES=24
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.65" # IP de bugatti
MASTER_PORT=12355

# Noms des machines (rank 0 en premier)
# bugatti est le master, suivi de 23 machines désignées par leur index/nom
RANK_NAMES=("bugatti" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23")

# --- Script d'entraînement parallélisé ---
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_spectrogram_parallel.py"

# --- Arguments du modèle / dataset ---
ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/regularized_window_double_exp/NSTATES=684_test_14_07_V2"
TARGET="vocals"
LR=0.004             # Taux d'apprentissage (learning rate, par exemple 0.001, ou plus grand si grand batch size)
EPOCHS=300
EPOCHS_RESUME=150    # Nombre d'époques supplémentaires à effectuer (150 epochs additionnelles)
BATCH_SIZE=8
NB_WORKERS=5
SEQ_DUR=4
CHUNK_DUR=1
NB_MAGSSM_STATES=684
N_FFT=682
N_HOP=34

# --- Régularisation L2 sur les parties imaginaires des valeurs propres ---
ALPHA=0          # facteur multiplicatif de la loss L2 (régularisation avec alpha=0.01)
BETA=0              # offset sur la pénalité au-dessus de pi (0 = bord franc)

# --- Paramètres de stabilité / initialisation du SSM ---
EPS_STABILITY=0     # marge de stabilité eps pour Progressive_SSM (0 = tighter)
DT_MIN=0.001        # timescale minimale du log-step (défaut 1e-3)
DT_MAX=0.1          # timescale maximale du log-step (défaut 0.1)

# Options de contraintes sur les valeurs propres réelles :
# 'abs' | 'relu' | 'sigmoid_interval' (reparamétrisation differentiable)
ENSURE_STABILITY="relu"
RE_LOWER=-0.0005  # -5 * 10^(-4)
RE_UPPER=-0.0000005  # -5 * 10^(-7)
SIGMOID_SCALE=1.0  # multiplicateur interne de gradient de la sigmoïde (défaut 1.0)


# --- Flags booléens (mettre à 1 pour activer, 0 pour désactiver) ---
FLAG_IS_WAV=1
FLAG_MEL=0
FLAG_AMP=0
FLAG_OG=1           # initialisation originale MagSSM

# ─────────────────────────────────────────────────────────────────────
# Configuration de la fenêtre à reprendre
# Format : "epsilon1 lambda_coeff_1 lambda_coeff_2"
# ─────────────────────────────────────────────────────────────────────
WINDOW_CONFIGS=(
    # "0.5000   4.7027   4.0000" #-30
    "0 0 0"  
)
