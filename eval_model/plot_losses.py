import matplotlib.pyplot as plt
import json
import numpy as np



dic = {
        "s512h512" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/magssm_fixed_init/s512h512/vocals.json",
        "s256h512" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/magssm_fixed_init/s256h512/vocals.json",
        "MagSSM-STFT-128bins" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/trainable_spectrograms/128bins/vocals.json"
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