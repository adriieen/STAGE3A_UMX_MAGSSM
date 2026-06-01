import matplotlib.pyplot as plt
import json
import numpy as np



dic = {
    "UNID" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/1d/vocals.json",
    "BILSTM" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/bilstm_classic/vocals.json",
    "UMX-Edge" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/sedge/vocals.json",
    "UMX-Edge-var" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/sedge_var/vocals.json",
    "Test : Edge+ a sort of MAGSSM" : "/home/adubois/openunmix/OpenUnmix/outputs/failed_models/magssm/vocals.json",
    "FALSE -- Edge+MAGSSM-48 states" : "/home/adubois/openunmix/OpenUnmix/outputs/failed_models/true_magssm/48states/hid96/vocals.json",
    "FALSE -- Edge+MAGSSM-96states (hsize = 256)" : "/home/adubois/openunmix/OpenUnmix/outputs/failed_models/true_magssm/96states/hid256/vocals.json",
    "FALSE -- Edge+MAGSSM-256states (hsize = 128)" : "/home/adubois/openunmix/OpenUnmix/outputs/failed_models/true_magssm/256states/hid128/vocals.json",
    "FALSE -- Edge+MAGSSM-256states (hsize = 512)" : "/home/adubois/openunmix/OpenUnmix/outputs/failed_models/true_magssm/256states/hid512/vocals.json",
    "Edge+MAGSSM - 256 states (hsize = 128) - nfft = 2048" : "/home/adubois/openunmix/OpenUnmix/outputs/failed_models/true_magssm/256states/hid128_fixed_init/vocals.json",
    "Edge+MAGSSM - 256 states (hsize = 512)" : "/home/adubois/openunmix/OpenUnmix/outputs/failed_models/true_magssm/256states/hid512_fixed_init/vocals.json",
    "Edge+MAGSSM - 256 states (hsize = 256)" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/magssm_fixed_init/256states_hid256/vocals.json"

}

for method, path in dic.items(): 
    with open(path, 'r') as data_json:
        data = json.load(data_json)

    train_loss = data["train_loss_history"]
    valid_loss = data["valid_loss_history"]
    best_ep = data["best_epoch"]
    best_l = data["best_loss"]

    # Créer une nouvelle figure propre à chaque itération
    plt.figure(figsize=(10, 6))

    epochs = np.arange(len(valid_loss))
    plt.plot(epochs, valid_loss, color='blue', label='Valid Loss')
    plt.plot(epochs, train_loss, color='red', label='Train Loss')

    # Tracer la ligne verticale pour la meilleure époque
    plt.axvline(x=best_ep, color='green', linestyle='--', 
                label=f'Best epoch {best_ep}, loss = {best_l:.4f}')

    plt.legend()
    plt.grid(True) # Ajouter une grille pour mieux lire l'ordonnée
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')

    plt.title(f"Train and validation losses : {method}")

    plt.savefig(f'/home/adubois/openunmix/OpenUnmix/fig/{method}.png')
    plt.close() # Fermer la figure pour libérer la mémoire