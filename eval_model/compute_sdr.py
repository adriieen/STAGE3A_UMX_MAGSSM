import os
from pathlib import Path
import json
import numpy as np

dic = {
     "UNID" : "/home/adubois/openunmix/OpenUnmix/evals/500_epochs/1d/test",
    # "BILSTM" : "/home/adubois/openunmix/OpenUnmix/evals/100_epoch/2d/test",
    "UMX-Edge" : "/home/adubois/openunmix/OpenUnmix/evals/500_epochs/Sedge/test"
}

for method in dic:
    path = Path(dic[method])
    # Accumulateur pour tous les scores de chaque frame de chaque chanson
    track_medians = {"SDR": [], "SIR": [], "SAR": [], "ISR": []}        
    print(f"--- Method: {method} ---")
    
    file_count = 0
    for root, folders, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as data_json:
                    data = json.load(data_json)
                
                # Extraction des segments valides (frames de 1s)
                clean_metrics = [
                    f["metrics"] for f in data["targets"][0]["frames"] 
                    if not any(map(lambda x: x is None or str(x) == 'nan', f["metrics"].values()))
                ]

                if clean_metrics:
                    file_count += 1
                    file_med = {}
                    for score in ["SDR", "SIR", "SAR", "ISR"]:
                        # Médiane des frames pour cette chanson précise
                        values = [m[score] for m in clean_metrics]
                        med_val = np.median(values)
                        file_med[score] = med_val
                        
                        # Accumulation de chaque frame individuelle pour la médiane globale
                        track_medians[score].append(med_val)
                    
                    print(f"File {file_count:02d}: {file[:30]}... | SDR Median: {file_med['SDR']:.3f} | SIR Median: {file_med['SIR']:.3f}")

    
    final_scores = {s: np.median(vals) if vals else 0 for s, vals in track_medians.items()}
    print(f"\n" + "="*45)
    print(f"{'METRIC':<10} | {'VALUE (Median of Medians)':<25}")
    print("-" * 45)

    for score, val in final_scores.items():
        # :<10 aligne le texte à gauche sur 10 caractères
        # :.3f force l'affichage de 3 chiffres après la virgule
        print(f"{score:<10} | {val:>15.3f}")

    print("="*45 + "\n")