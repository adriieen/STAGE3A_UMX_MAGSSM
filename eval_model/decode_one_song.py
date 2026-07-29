import json
import os
import sys
from pathlib import Path

import torch
import torchaudio

from path_config import setup_paths
setup_paths()

from decoder import Trainable_decoder
from transforms import get_regularised_window

# Paths
model_path = "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_decoders/[8F]spectrogram_loss_modified_ssm/eps0.5000_l1_4.7027_l2_4.0000/vocals.pth"
root_path = "/Data/adrien.dubois/musdb18_ds3"
wav_save_path = "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/audio_samples"

os.makedirs(wav_save_path, exist_ok=True)

# 1. Lecture dynamique des hyperparamètres depuis separator.json
model_dir = Path(model_path).parent
separator_json_path = model_dir / "separator.json"

if not separator_json_path.exists():
    raise FileNotFoundError(f"Fichier de configuration introuvable : {separator_json_path}")

print(f"Lecture des hyperparamètres depuis : {separator_json_path}")
with open(separator_json_path, "r") as f:
    config = json.load(f)

n_fft = config.get("nfft", config.get("n_fft", 80))
n_hop = config.get("nhop", config.get("n_hop", 4))
dim_state = config.get("nb_magssm_states", 328)
target_sr = int(config.get("sample_rate", 14700))
duration_sec = 30.0

epsilon1 = config.get("epsilon1", 0.5)
lambda_coeff_1 = config.get("lambda_coeff_1", 4.7027)
lambda_coeff_2 = config.get("lambda_coeff_2", 4.0)
regularize_window = config.get("regularize_window", True)

print(f"Configuration chargée : n_fft={n_fft}, n_hop={n_hop}, dim_state={dim_state}, sample_rate={target_sr} Hz")

# 2. Chargement des 30 premières secondes de la première chanson du test set
print(f"\n--- Chargement des 30 premières secondes de la première chanson du test set ---")
test_dir = Path(root_path) / "test"
first_track_dir = sorted([d for d in test_dir.iterdir() if d.is_dir()])[0]
vocals_wav_path = first_track_dir / "vocals.wav"

print(f"Morceau sélectionné : {first_track_dir.name}")
print(f"Fichier vocal source : {vocals_wav_path}")

info = torchaudio.info(str(vocals_wav_path))
num_frames_to_load = int(duration_sec * info.sample_rate)

audio, orig_sr = torchaudio.load(str(vocals_wav_path), num_frames=num_frames_to_load)

# Ré-échantillonnage vers target_sr si nécessaire
if orig_sr != target_sr:
    print(f"Ré-échantillonnage de {orig_sr} Hz vers {target_sr} Hz...")
    audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=target_sr)

# Assurer la forme Batch (B=1, C=2, N_samples)
if audio.ndim == 2:
    audio = audio.unsqueeze(0)

B, C, N_samples = audio.shape
print(f"Signal temporel audio chargé : Forme = {audio.shape}, Fréquence = {target_sr} Hz")

# 3. Construction de la fenêtre et Encodage STFT
print(f"\n--- Construction de la fenêtre et Encodage STFT ---")
if regularize_window:
    window = get_regularised_window(
        n_fft=n_fft,
        epsilon1=epsilon1,
        lambda_coeff_1=lambda_coeff_1,
        lambda_coeff_2=lambda_coeff_2,
        device=audio.device,
        dtype=torch.float32
    )
else:
    window = torch.hann_window(n_fft, device=audio.device, dtype=torch.float32)

stft_channels = []
for c in range(C):
    stft_c = torch.stft(
        audio[:, c, :],
        n_fft=n_fft,
        hop_length=n_hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True
    )
    stft_channels.append(stft_c)

X_complex = torch.stack(stft_channels, dim=1) # (B=1, C=2, F, T) complexe
print(f"Spectrogramme complexe obtenu : Forme = {X_complex.shape}")

# 4. Chargement du Décodeur Entraîné
print(f"\n--- Chargement du Décodeur Entraîné ---")
decoder = Trainable_decoder(
    sample_rate=target_sr,
    n_fft=n_fft,
    n_hop=n_hop,
    length=N_samples,
    window=window,
    dim_state=dim_state,
    structured_initialisation=True,
    C_C_init="istft"
)

state_dict = torch.load(model_path, map_location="cpu")
decoder.load_state_dict(state_dict)
decoder.eval()
print(f"Poids du modèle chargés depuis : {model_path}")

# 5. Décodage du Spectrogramme vers le Signal Temporel
print(f"\n--- Décodage du Spectrogramme vers le Signal Temporel ---")
with torch.no_grad():
    y_decoded = decoder(X_complex, length=N_samples)

print(f"Signal temporel décodé : Forme = {y_decoded.shape}")

# 6. Sauvegarde du fichier .wav
print(f"\n--- Sauvegarde du fichier .wav ---")
output_wav_file = os.path.join(wav_save_path, f"{first_track_dir.name}_decoded_vocals.wav")
y_wav = y_decoded.squeeze(0).cpu()

torchaudio.save(output_wav_file, y_wav, sample_rate=target_sr)
print(f"✅ Fichier audio décodé enregistré avec succès : {output_wav_file}")
