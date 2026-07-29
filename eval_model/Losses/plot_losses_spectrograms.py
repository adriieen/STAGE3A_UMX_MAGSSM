import matplotlib.pyplot as plt
import json
import numpy as np
import os

plot_lambdas_re = False
plot_lambdas_im = True

dic = {
  
    "[MAG] DS3 : -30dB" : "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/og_magssm_non_progressive_170bins/eps0.5000_l1_4.7027_l2_4.0000/vocals.json",
    "[CPLX] DS3 : -30dB, free lambda, default init" : "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/non_progressive_170bins_alpha0_default_init/eps0.5000_l1_4.7027_l2_4.0000/vocals.json",
    "[CPLX] DS3 : -30dB, free lambda, structured init" : "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/non_progressive_170bins_alpha0_structured_init/eps0.5000_l1_4.7027_l2_4.0000/vocals.json",
    "[CPLX] DS3 : -30dB, regularized lambda, default init" : "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/non_progressive_170bins_alpha4e-3/eps0.5000_l1_4.7027_l2_4.0000/vocals.json",
    "[CPLX] DS3 : -30dB, regularized lambda, structured init" : "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/non_progressive_170bins_alpha4e-3_structured_init_bis/eps0.5000_l1_4.7027_l2_4.0000/vocals.json",
    "[CPLX] DS3 : -30dB, regularized lambda, structured init, 3F": "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_spectograms/14.7kHz/Tests_post_soutenance/complex_specto/[3F]non_progressive_170bins_alpha4e-3_structured_init_bis/eps0.5000_l1_4.7027_l2_4.0000/vocals.json",
    "[DECODER] -30 dB, regularized lambda, strucured init, v0" : "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/trainable_decoders/eps0.5000_l1_4.7027_l2_4.0000/vocals.json"
}





















for method, path in dic.items(): 
    with open(path, 'r') as data_json:
        data = json.load(data_json)

    train_loss = data["train_loss_history"]
    valid_loss = data["valid_loss_history"]
    best_ep = data["best_epoch"]
    best_l = data["best_loss"]

    # Créer une nouvelle figure propre à chaque itération
    fig, ax1 = plt.subplots(figsize=(11, 6))

    epochs = np.arange(1, len(valid_loss)+1)
    ax1.plot(epochs, valid_loss, color='blue', label='Valid Loss')
    ax1.plot(epochs, train_loss, color='red', label='Train Loss')

    # Tracer la ligne verticale pour la meilleure époque
    ax1.axvline(x=best_ep, color='green', linestyle='--', 
                label=f'Best epoch {best_ep}, loss = {best_l:.4f}')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)')
    # ax1.set_ylim(0,2)
    ax1.grid(True) # Ajouter une grille pour mieux lire l'ordonnée

    legend_axes = [ax1]
    lambda_path = os.path.join(os.path.dirname(path), 'vocals_lambda.json')
    if (plot_lambdas_re or plot_lambdas_im) and os.path.exists(lambda_path):
        try:
            with open(lambda_path, 'r') as lambda_json:
                lambda_data = json.load(lambda_json)

            lambda_epochs = []
            re_min = []
            re_max = []
            im_min = []
            im_max = []
            im_q1 = []
            im_q3 = []

            for entry in lambda_data:
                ep = entry.get("epoch")
                ld_re_mins = []
                ld_re_maxs = []
                ld_im_mins = []
                ld_im_maxs = []
                ld_im_q1 = []
                ld_im_q3 = []

                for module, stats in entry.get("lambda", {}).items():
                    if "ld_re_min" in stats:
                        ld_re_mins.append(stats["ld_re_min"])
                    if "ld_re_max" in stats:
                        ld_re_maxs.append(stats["ld_re_max"])
                    if "ld_im_min" in stats:
                        ld_im_mins.append(stats["ld_im_min"])
                    if "ld_im_max" in stats:
                        ld_im_maxs.append(stats["ld_im_max"])
                    if "ld_im_q1" in stats:
                        ld_im_q1.append(stats["ld_im_q1"])
                    if "ld_im_q3" in stats:
                        ld_im_q3.append(stats["ld_im_q3"])

                if ld_re_mins and ld_re_maxs and ld_im_mins and ld_im_maxs:
                    lambda_epochs.append(ep)
                    re_min.append(min(ld_re_mins))
                    re_max.append(max(ld_re_maxs))
                    im_min.append(min(ld_im_mins))
                    im_max.append(max(ld_im_maxs))
                    if ld_im_q1 and ld_im_q3:
                        im_q1.append(min(ld_im_q1))
                        im_q3.append(max(ld_im_q3))

            im_min, im_max = np.array(im_min)/np.pi, np.array(im_max)/np.pi
            if len(im_q1) == len(lambda_epochs) and len(im_q3) == len(lambda_epochs):
                im_q1 = np.array(im_q1)/np.pi
                im_q3 = np.array(im_q3)/np.pi
            else:
                im_q1, im_q3 = [], []

            if lambda_epochs:
                # Count how many we are actually plotting
                num_right_axes = 0
                if plot_lambdas_re:
                    num_right_axes += 1
                if plot_lambdas_im:
                    num_right_axes += 1

                if num_right_axes == 2:
                    fig.subplots_adjust(right=0.75)
                    ax2 = ax1.twinx()
                    ax3 = ax1.twinx()
                    ax3.spines['right'].set_position(('outward', 60))

                    # Tracer les bandes min/max (λ_re ∈ R^-, on trace -λ_re > 0 en log)
                    ax2.fill_between(lambda_epochs, -np.array(re_max), -np.array(re_min), color='orange', alpha=0.15, label=r'$-\lambda_{re}$ min/max')
                    ax2.plot(lambda_epochs, -np.array(re_max), color='orange', linestyle=':', linewidth=1)
                    ax2.plot(lambda_epochs, -np.array(re_min), color='orange', linestyle=':', linewidth=1)
                    ax2.set_yscale('log')
                    ax2.set_ylabel(r'$-\lambda_{re}$', color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')

                    ax3.fill_between(lambda_epochs, im_min, im_max, color='purple', alpha=0.15, label=r'$\lambda_{im}$ min/max')
                    ax3.plot(lambda_epochs, im_min, color='purple', linestyle=':', linewidth=1)
                    ax3.plot(lambda_epochs, im_max, color='purple', linestyle=':', linewidth=1)
                    if len(im_q1) > 0 and len(im_q3) > 0:
                        ax3.fill_between(lambda_epochs, im_q1, im_q3, color='purple', alpha=0.3, label=r'$\lambda_{im}$ Q1/Q3')
                        ax3.plot(lambda_epochs, im_q1, color='purple', linestyle='--', linewidth=1)
                        ax3.plot(lambda_epochs, im_q3, color='purple', linestyle='--', linewidth=1)
                    ax3.axhline(y=1.0, color='purple', linestyle=':', linewidth=1, label=r'$\lambda_{im} = \pi$')
                    ax3.set_ylabel(r'$\lambda_{im}/\pi$', color='purple')
                    ax3.tick_params(axis='y', labelcolor='purple')

                    legend_axes = [ax1, ax2, ax3]
                elif num_right_axes == 1:
                    fig.subplots_adjust(right=0.85)
                    ax2 = ax1.twinx()
                    if plot_lambdas_re:
                        ax2.fill_between(lambda_epochs, -np.array(re_max), -np.array(re_min), color='orange', alpha=0.15, label=r'$-\lambda_{re}$ min/max')
                        ax2.plot(lambda_epochs, -np.array(re_max), color='orange', linestyle=':', linewidth=1)
                        ax2.plot(lambda_epochs, -np.array(re_min), color='orange', linestyle=':', linewidth=1)
                        ax2.set_yscale('log')
                        ax2.set_ylabel(r'$-\lambda_{re}$', color='orange')
                        ax2.tick_params(axis='y', labelcolor='orange')
                    else: # plot_lambdas_im
                        ax2.fill_between(lambda_epochs, im_min, im_max, color='purple', alpha=0.15, label=r'$\lambda_{im}$ min/max')
                        ax2.plot(lambda_epochs, im_min, color='purple', linestyle=':', linewidth=1)
                        ax2.plot(lambda_epochs, im_max, color='purple', linestyle=':', linewidth=1)
                        if len(im_q1) > 0 and len(im_q3) > 0:
                            ax2.fill_between(lambda_epochs, im_q1, im_q3, color='purple', alpha=0.3, label=r'$\lambda_{im}$ Q1/Q3')
                            ax2.plot(lambda_epochs, im_q1, color='purple', linestyle='--', linewidth=1)
                            ax2.plot(lambda_epochs, im_q3, color='purple', linestyle='--', linewidth=1)
                        ax2.axhline(y=1.0, color='purple', linestyle=':', linewidth=1, label=r'$\lambda_{im} = \pi$')
                        ax2.set_ylabel(r'$\lambda_{im}/\pi$', color='purple')
                        ax2.tick_params(axis='y', labelcolor='purple')

                    legend_axes = [ax1, ax2]
        except Exception as e:
            print(f"Error plotting lambda for {method}: {e}")

    # Gérer la légende combinée
    handles = []
    labels = []
    for ax in legend_axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax1.legend(handles, labels, loc='upper left')
    ax1.set_ylim(0,0.05)
    plt.title(f"Train and validation losses : {method}")

    # S'assurer que le dossier fig existe
    os.makedirs('/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fig/trainable_spectrograms/post_soutenance', exist_ok=True)
    plt.savefig(f"/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fig/trainable_spectrograms/post_soutenance/{method}.png")
    plt.close() # Fermer la figure pour libérer la mémoire