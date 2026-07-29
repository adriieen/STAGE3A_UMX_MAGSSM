#!/usr/bin/env bash
"""
Script d'extraction des 3 grandes briques d'un modèle finetuné combiné :
1. get_c_magssm  : Extraie les poids du Trainable Spectrogram (C-MAGSSM Encoder)
2. get_dec_ssm    : Extraie les poids du Trainable Decoder (DEC-SSM Decoder)
3. get_umx        : Extraie les poids du Backbone OpenUnmix (Bi-LSTM + couches FC)

Peut être importé comme module Python ou exécuté directement en CLI.
"""

import os
import argparse
from pathlib import Path
import torch


def get_c_magssm(checkpoint_path: str, output_path: str = None) -> dict:
    """Extraie les poids du C-MAGSSM (Trainable Spectrogram) d'un checkpoint finetuné.

    Supprime le préfixe 'trainable_spectrogram.' des clés du state_dict pour que
    les poids puissent être directement chargés dans un module Trainable_spectrogram.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "state_dict_model" in ckpt:
            model_state = ckpt["state_dict_model"]
        elif "state_dict" in ckpt:
            model_state = ckpt["state_dict"]
        else:
            model_state = ckpt
    else:
        model_state = ckpt

    cmagssm_state = {}
    for key, value in model_state.items():
        if key.startswith("trainable_spectrogram."):
            new_key = key.replace("trainable_spectrogram.", "", 1)
            cmagssm_state[new_key] = value

    if not cmagssm_state:
        raise ValueError(f"Aucune clé 'trainable_spectrogram.' trouvée dans {checkpoint_path}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(cmagssm_state, output_path)
        print(f"✓ Poids C-MAGSSM sauvegardés dans : {output_path}")

    return cmagssm_state


def get_dec_ssm(checkpoint_path: str, output_path: str = None) -> dict:
    """Extraie les poids du DEC-SSM (Trainable Decoder) d'un checkpoint finetuné."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    dec_state = None
    if isinstance(ckpt, dict) and "state_dict_decoder" in ckpt:
        dec_state = ckpt["state_dict_decoder"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and not any(k.startswith("trainable_spectrogram.") for k in ckpt["state_dict"]):
        dec_state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "state_dict_model" not in ckpt and "state_dict" not in ckpt:
        dec_state = ckpt
    else:
        # Recherche d'un fichier _decoder.pth adjacent
        p = Path(checkpoint_path)
        stem = p.name.split(".")[0]
        dec_file = p.parent / f"{stem}_decoder.pth"
        if dec_file.exists():
            print(f"✓ Fichier décodeur trouvé : {dec_file}")
            dec_ckpt = torch.load(dec_file, map_location="cpu", weights_only=False)
            dec_state = dec_ckpt.get("state_dict", dec_ckpt)
        else:
            raise KeyError(f"Impossible de trouver 'state_dict_decoder' dans {checkpoint_path} ni de fichier {dec_file}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(dec_state, output_path)
        print(f"✓ Poids DEC-SSM sauvegardés dans : {output_path}")

    return dec_state


def get_umx(checkpoint_path: str, output_path: str = None) -> dict:
    """Extraie les poids du Backbone OpenUnmix (Bi-LSTM + couches FC) d'un checkpoint finetuné.

    Exclut les poids du C-MAGSSM encoder pour ne conserver que le réseau OpenUnmix (fc1, ln1, lstm, fc2, ln2, fc3, etc.).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "state_dict_model" in ckpt:
            model_state = ckpt["state_dict_model"]
        elif "state_dict" in ckpt:
            model_state = ckpt["state_dict"]
        else:
            model_state = ckpt
    else:
        model_state = ckpt

    umx_state = {}
    for key, value in model_state.items():
        if not key.startswith("trainable_spectrogram."):
            umx_state[key] = value

    if not umx_state:
        raise ValueError(f"Aucune clé du backbone OpenUnmix trouvée dans {checkpoint_path}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(umx_state, output_path)
        print(f"✓ Poids OpenUnmix backbone sauvegardés dans : {output_path}")

    return umx_state


def extract_all_blocks(checkpoint_path: str, output_dir: str, target: str = "vocals"):
    """Extraie les 3 briques (C-MAGSSM, DEC-SSM, UMX) dans un dossier donné."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmagssm_out = out_dir / f"{target}_c_magssm.pth"
    dec_out = out_dir / f"{target}_dec_ssm.pth"
    umx_out = out_dir / f"{target}_umx.pth"

    print(f"=== Extraction des 3 briques depuis : {checkpoint_path} ===")
    get_c_magssm(checkpoint_path, str(cmagssm_out))
    get_dec_ssm(checkpoint_path, str(dec_out))
    get_umx(checkpoint_path, str(umx_out))
    print(f"✓ Extraction terminée avec succès dans : {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extraire les 3 grandes briques (.pth) d'un checkpoint combiné finetuné.")
    parser.add_argument("checkpoint", type=str, help="Chemin du checkpoint (.chkpnt ou .pth)")
    parser.add_argument("--output-dir", type=str, default="./extracted_models", help="Dossier de destination")
    parser.add_argument("--target", type=str, default="vocals", help="Nom de la cible (ex: vocals)")
    parser.add_argument("--c-magssm-out", type=str, default=None, help="Chemin spécifique de sortie pour C-MAGSSM .pth")
    parser.add_argument("--dec-ssm-out", type=str, default=None, help="Chemin spécifique de sortie pour DEC-SSM .pth")
    parser.add_argument("--umx-out", type=str, default=None, help="Chemin spécifique de sortie pour OpenUnmix .pth")

    args = parser.parse_args()

    if args.c_magssm_out or args.dec_ssm_out or args.umx_out:
        if args.c_magssm_out:
            get_c_magssm(args.checkpoint, args.c_magssm_out)
        if args.dec_ssm_out:
            get_dec_ssm(args.checkpoint, args.dec_ssm_out)
        if args.umx_out:
            get_umx(args.checkpoint, args.umx_out)
    else:
        extract_all_blocks(args.checkpoint, args.output_dir, target=args.target)


if __name__ == "__main__":
    main()
