#!/usr/bin/env bash
# ============================================================
#  CONFIG CENTRALE FINE-TUNING — COMBINED PIPELINE
#  C-MAGSSM (Encoder) + OpenUnmix (Backbone) + DEC-SSM (Decoder)
#  Modifier ICI, puis lancer `bash generate_ranks_sweep.sh`
# ============================================================

# ============================================================
#  PATHS DES MODÈLES PRÉ-ENTRAÎNÉS (.pth / dossier checkpoint)
# ============================================================
# Indiquer ici les chemins vers les modèles pré-entraînés.
# Si laissé vide "", le composant sera initialisé aléatoirement / par défaut.

# 1. Chemins du modèle Trainable Spectrogram (C-MAGSSM Encoder)
SPECTROGRAM_MODEL="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/[small_lr]non_progressive_342bins_alpha0_structured_init/eps0.5000_l1_4.7027_l2_4.0000"   # path to folder containing .pth 

# 2. Chemins du modèle Trainable Decoder (DEC-SSM Decoder)
DECODER_MODEL="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_decoders/[341bins_modified_nhop_low_lr]spectrogram_loss_modified_ssm/eps0.5000_l1_4.7027_l2_4.0000"        # path to folder containing .pth 

# 3. Chemins du modèle OpenUnmix (Backbone : LSTM + couches FC)
UMX_MODEL="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/post_soutenance/umx_training/[28-07]seed=42/eps0.5000_l1_4.7027_l2_4.0000"            # path to folder containing .pth 

# 4. Checkpoint global pour reprise d'entraînement du pipeline combiné
CHECKPOINT=""          # ex: "/path/to/combined_checkpoint_dir" (reprise automatique si vide mais chkpnt présent dans output_dir)


# ============================================================
#  TOPOLOGIE DDP MULTI-MACHINES
# ============================================================
NNODES=18
NPROC_PER_NODE=1
MASTER_ADDR="129.104.252.65"
MASTER_PORT=12355

# Noms des machines (rank 0 en premier). Longueur égale à NNODES
RANK_NAMES=("bugatti" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17")


# ============================================================
#  SCRIPT D'ENTRAÎNEMENT & CONDA ENVIRONMENT
# ============================================================
TRAIN_SCRIPT="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/open-unmix-pytorch/openunmix/train_magssm_parallel.py"
TORCHRUN_BIN="/users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/torchrun"


# ============================================================
#  PARAMÈTRES DE DATASET & HYPERPARAMÈTRES D'ENTRAÎNEMENT
# ============================================================
# NOTE : Les paramètres d'architecture du modèle (nb_magssm_states, nfft, nhop,
# d_out, og, structured_initialisation, regularize_window, etc.) sont 
# AUTOMATIQUEMENT DÉTECTÉS et VÉRIFIÉS à partir des modèles pré-entraînés fournis 
# (SPECTROGRAM_MODEL et DECODER_MODEL). Vous n'avez pas besoin de les redéfinir ici !

ROOT="/Data/adrien.dubois/musdb18_ds3"
OUTPUT_BASE="/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/fine_tuning_july/vocals_pipeline/joint_optimization"
TARGET="vocals"

EPOCHS=400
EPOCHS_RESUME=40       # Époques supplémentaires si reprise
BATCH_SIZE=6
NB_WORKERS=5
SEQ_DUR=4
CHUNK_DUR=1

# Learning Rates
LEARNING_RATE=0.0005       # LR pour les composants SSM (Encoder / Decoder)
LR_BACKBONE=0.001       # LR pour le backbone OpenUnmix (si non gelé)


# ============================================================
#  GEL DE COUCHES (FREEZING)
# ============================================================
# 1 = activé (geler les poids), 0 = désactivé (co-optimiser / entraîner)
FLAG_FREEZE_BACKBONE=0    # 1 = Geler le backbone OpenUnmix, 0 = Co-optimiser OpenUnmix
FLAG_FREEZE_SSM=0         # 1 = Geler C-MAGSSM Spectrogram SSM
FLAG_FREEZE_DECODER=0     # 1 = Geler DEC-SSM Decoder


# ============================================================
#  RÉGULARISATION DE LOSS & STABILITÉ SSM
# ============================================================
ALPHA=0                   # Facteur multiplicatif loss L2_im_lambda pendant le fine-tuning (0 = pas de loss L2)
BETA=0                    # Offset fréquence au-dessus de pi (0 = bord franc)

# NOTE : Les paramètres de stabilité et d'initialisation du SSM (eps_stability, 
# dt_min, dt_max, og, structured_initialisation, mel, phase_correction, etc.)
# sont hérité(e)s AUTOMATIQUEMENT des modèles pré-entraînés fournis.
EPS_STABILITY=0           # Marge de stabilité (hérité si non spécifié)
DT_MIN=0.001              # Timescale minimale log-step
DT_MAX=0.1                # Timescale maximale log-step


# ============================================================
#  FLAGS BOOLÉENS (1 = Oui, 0 = Non)
# ============================================================
FLAG_PROGRESSIVE=0        # 1 = Traitement par chunks (économie VRAM)
FLAG_FFT_KERNEL=1         # 1 = Noyau FFT parallèle (accélération GPU)
FLAG_AMP=0                # 1 = Automatic Mixed Precision (fp16/bf16)

# NOTE : complex_spectrogram=1 est automatiquement VÉRIFIÉ et EXIGÉ pendant le 
# training par train_magssm_parallel.py (erreur levée si le modèle pré-entraîné est en magnitude seule).
FLAG_COMPLEX_SPECTROGRAM=1
FLAG_OG=1
FLAG_STRUCTURED_INITIALISATION=1
FLAG_PHASE_CORRECTION=0


# ============================================================
#  PARAMÈTRES DE FENÊTRE À BALAYER (SWEEP)
#  Format : "epsilon1 lambda_coeff_1 lambda_coeff_2"
# ============================================================
WINDOW_CONFIGS=(
    "0.5000   4.7027   4.0000" # -30dB
)
