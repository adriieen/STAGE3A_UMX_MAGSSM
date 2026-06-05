"""
path_config.py — Chargement des chemins machine-spécifiques depuis machine.yaml

Le fichier machine.yaml est placé à la racine du projet et est ignoré par git.
Il contient les chemins propres à chaque machine (voir machine.yaml.example).

Usage dans un module Python :
    from path_config import setup_paths, amp_autocast, amp_grad_scaler
    setup_paths()   # ajoute sedge_src (et optionnellement openunmix_src) au sys.path

    # Autocast portable (Python 3.7/torch ancien OU Python 3.10/torch récent) :
    with amp_autocast(enabled=True):
        ...

    # GradScaler portable :
    scaler = amp_grad_scaler(enabled=True)

Le fichier machine.yaml est cherché à la racine du dépôt (deux niveaux au-dessus
de ce fichier : OpenUnmix/open-unmix-pytorch/openunmix/ → OpenUnmix/).

Champ optionnel dans machine.yaml :
    pytorch_env: "3.10"   # force l'API torch.amp.autocast("cuda", ...)
    pytorch_env: "3.7"    # force l'API torch.cuda.amp.autocast(...)
    # (absent → auto-détection via la version de PyTorch installée)
"""

import sys
import contextlib
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError(
        "Le module 'pyyaml' est requis pour charger machine.yaml. "
        "Installe-le avec : pip install pyyaml"
    )

import torch


# ---------------------------------------------------------------------------
# Détection de l'API AMP disponible
# ---------------------------------------------------------------------------

def _detect_new_amp_api(config: dict) -> bool:
    """
    Renvoie True si on doit utiliser la nouvelle API torch.amp.autocast("cuda", ...)
    (PyTorch >= 1.10, typiquement Python 3.10 dans ton setup).

    Ordre de priorité :
      1. Champ `pytorch_env` dans machine.yaml  ("3.10" → nouvelle, "3.7" → ancienne)
      2. Auto-détection via torch.__version__
    """
    pytorch_env = config.get("pytorch_env", None)

    if pytorch_env is not None:
        env_str = str(pytorch_env).strip()
        if env_str == "3.10":
            return True
        elif env_str == "3.7":
            return False
        else:
            raise ValueError(
                f"Valeur inconnue pour 'pytorch_env' dans machine.yaml : {env_str!r}. "
                "Valeurs acceptées : \"3.7\" ou \"3.10\"."
            )

    # Auto-détection : torch.amp.autocast existe depuis PyTorch 1.10
    return hasattr(torch.amp, "autocast")


# ---------------------------------------------------------------------------
# Helpers AMP portables — à importer à la place de torch.cuda.amp.*
# ---------------------------------------------------------------------------

# Ces variables sont initialisées après le premier appel à setup_paths().
_USE_NEW_AMP_API: bool = False
_AMP_INITIALIZED: bool = False


def amp_autocast(enabled: bool = True, device_type: str = "cuda"):
    """
    Context manager AMP portable.

    Equivalent de :
      - (ancienne API, Python 3.7 / PyTorch < 1.10) :
            torch.cuda.amp.autocast(enabled=enabled)
      - (nouvelle API, Python 3.10 / PyTorch >= 1.10) :
            torch.amp.autocast(device_type, enabled=enabled)

    Usage :
        with amp_autocast(enabled=use_amp):
            ...
        # Pour désactiver explicitement (ex: dans le forward du modèle) :
        with amp_autocast(enabled=False):
            ...
    """
    if not _AMP_INITIALIZED:
        # Fallback si appelé avant setup_paths() — auto-détection directe
        use_new = hasattr(torch.amp, "autocast")
    else:
        use_new = _USE_NEW_AMP_API

    if use_new:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled=enabled)


def amp_grad_scaler(enabled: bool = True):
    """
    Crée un GradScaler portable.

    Equivalent de :
      - (ancienne API) : torch.cuda.amp.GradScaler(enabled=enabled)
      - (nouvelle API) : torch.amp.GradScaler("cuda", enabled=enabled)

    Usage :
        scaler = amp_grad_scaler(enabled=args.amp and use_cuda)
    """
    if not _AMP_INITIALIZED:
        use_new = hasattr(torch.amp, "GradScaler")
    else:
        use_new = _USE_NEW_AMP_API

    if use_new and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    else:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Chargement du fichier machine.yaml
# ---------------------------------------------------------------------------

def _find_machine_yaml() -> Path:
    """Remonte l'arborescence depuis ce fichier pour trouver machine.yaml."""
    # Ce fichier est dans OpenUnmix/open-unmix-pytorch/openunmix/
    # machine.yaml est dans OpenUnmix/
    current = Path(__file__).resolve().parent
    for _ in range(5):  # on remonte au plus 5 niveaux
        candidate = current / "machine.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "Fichier machine.yaml introuvable. "
        "Copie machine.yaml.example en machine.yaml à la racine du projet "
        "et adapte les chemins à ta machine."
    )


def setup_paths(add_openunmix_src: bool = False) -> dict:
    """
    Lit machine.yaml, ajoute les chemins nécessaires à sys.path,
    et configure les helpers AMP (amp_autocast / amp_grad_scaler).

    Args:
        add_openunmix_src: si True, ajoute aussi openunmix_src au sys.path.

    Returns:
        dict: le contenu brut du fichier machine.yaml (pour usage éventuel).
    """
    global _USE_NEW_AMP_API, _AMP_INITIALIZED

    yaml_path = _find_machine_yaml()
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Chemins sys.path ---
    sedge_src = config.get("sedge_src")
    if sedge_src is None:
        raise KeyError(
            f"Clé 'sedge_src' manquante dans {yaml_path}. "
            "Consulte machine.yaml.example pour le format attendu."
        )

    sedge_src = str(sedge_src)
    if sedge_src not in sys.path:
        sys.path.insert(0, sedge_src)

    if add_openunmix_src:
        openunmix_src = config.get("openunmix_src")
        if openunmix_src is not None:
            openunmix_src = str(openunmix_src)
            if openunmix_src not in sys.path:
                sys.path.insert(0, openunmix_src)

    # --- Configuration AMP ---
    _USE_NEW_AMP_API = _detect_new_amp_api(config)
    _AMP_INITIALIZED = True

    amp_label = "nouvelle (torch.amp.*)" if _USE_NEW_AMP_API else "ancienne (torch.cuda.amp.*)"
    print(f"[path_config] AMP API : {amp_label} | sedge_src : {sedge_src}")

    return config
