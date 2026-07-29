"""
Structure de données pour stocker les résultats de la recherche de paramètres
optimaux de régularisation de fenêtre, indexés par (hop_factor, alpha_ola).

Usage:
    from results_store import ResultsStore

    store = ResultsStore()

    # Stocker un résultat
    store.set(hop_factor=40, alpha_ola=1e4,
              epsilon1=0.07, lambda_coeff_1=0.77, lambda_coeff_2=0.85,
              relative_error_db=-42.3)

    # Récupérer un résultat
    r = store.get(hop_factor=40, alpha_ola=1e4)
    print(r.epsilon1, r.lambda_coeff_1, r.lambda_coeff_2, r.relative_error_db)

    # Sauvegarder / charger (JSON)
    store.save("results.json")
    store = ResultsStore.load("results.json")

    # Afficher un tableau récapitulatif
    store.summary()
"""
import json
from collections import namedtuple
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Paramètres optimaux + erreur de modulation OLA pour une configuration donnée
OptimalParams = namedtuple("OptimalParams", [
    "epsilon1",          # poids de l'exponentielle rapide (epsilon2 = 1 - epsilon1)
    "lambda_coeff_1",    # coefficient tel que lambda_val_1 = lambda_coeff_1 / N
    "lambda_coeff_2",    # coefficient tel que lambda_val_2 = lambda_coeff_2 / N
    "relative_error_db", # erreur relative de modulation OLA (dB), cf. test_régul_window.py L40
])


class ResultsStore:
    """
    Stocke les résultats indexés par (hop_factor, alpha_ola).

    - hop_factor : int — tel que nhop = N // hop_factor  
    - alpha_ola  : float — poids de la pénalité OLA dans la loss 
    """

    # Valeurs par défaut des axes de balayage
    DEFAULT_HOP_FACTORS = [20]
    DEFAULT_ALPHA_OLA_VALUES = [2250e1]  

    def __init__(self):
        # dict[(hop_factor, alpha_ola)] -> OptimalParams
        self._data: Dict[Tuple[int, float], OptimalParams] = {}

    # ------------------------------------------------------------------ #
    # Accès                                                                #
    # ------------------------------------------------------------------ #
    def set(self, hop_factor: int, alpha_ola: float, *,
            epsilon1: float, lambda_coeff_1: float, lambda_coeff_2: float,
            relative_error_db: float):
        """Enregistre les paramètres optimaux pour (hop_factor, alpha_ola)."""
        self._data[(hop_factor, alpha_ola)] = OptimalParams(
            epsilon1=epsilon1,
            lambda_coeff_1=lambda_coeff_1,
            lambda_coeff_2=lambda_coeff_2,
            relative_error_db=relative_error_db,
        )

    def get(self, hop_factor: int, alpha_ola: float) -> Optional[OptimalParams]:
        """Retourne les paramètres optimaux ou None si pas encore calculé."""
        return self._data.get((hop_factor, alpha_ola))

    def keys(self):
        return self._data.keys()

    def __len__(self):
        return len(self._data)

    def __contains__(self, key: Tuple[int, float]) -> bool:
        return key in self._data

    # ------------------------------------------------------------------ #
    # Sérialisation JSON                                                   #
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]):
        """Sauvegarde en JSON. Les clés tuple sont sérialisées en string."""
        serializable = {}
        for (hf, alpha), params in self._data.items():
            key_str = f"{hf}_{alpha}"
            serializable[key_str] = params._asdict()
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ResultsStore":
        """Charge depuis un fichier JSON."""
        store = cls()
        with open(path) as f:
            raw = json.load(f)
        for key_str, params_dict in raw.items():
            hf_str, alpha_str = key_str.split("_", 1)
            hf = int(hf_str)
            alpha = float(alpha_str)
            store._data[(hf, alpha)] = OptimalParams(**params_dict)
        return store

    # ------------------------------------------------------------------ #
    # Affichage                                                            #
    # ------------------------------------------------------------------ #
    def summary(self, N: int = 512):
        """Affiche un tableau récapitulatif formaté."""
        if not self._data:
            print("(aucun résultat stocké)")
            return

        hop_factors = sorted({k[0] for k in self._data})
        alphas = sorted({k[1] for k in self._data})

        # En-tête
        header = f"{'hop_factor':>11} {'nhop':>6} {'alpha_ola':>10} │ {'eps1':>8} {'λ1/N':>8} {'λ2/N':>8} │ {'err_mod (dB)':>12}"
        print(header)
        print("─" * len(header))

        for hf in hop_factors:
            for alpha in alphas:
                p = self.get(hf, alpha)
                nhop = N // hf
                if p is not None:
                    print(f"{hf:>11} {nhop:>6} {alpha:>10.0f} │ "
                          f"{p.epsilon1:>8.4f} {p.lambda_coeff_1:>8.4f} {p.lambda_coeff_2:>8.4f} │ "
                          f"{p.relative_error_db:>12.2f}")
                else:
                    print(f"{hf:>11} {nhop:>6} {alpha:>10.0f} │ {'—':>8} {'—':>8} {'—':>8} │ {'—':>12}")
            print()
