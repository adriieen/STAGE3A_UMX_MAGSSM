import numpy as np
import matplotlib.pyplot as plt

N = 683
t = np.arange(N)

epsilon1 = 0.11
epsilon2 = 1-epsilon1
lambda_val_1 = 0.1/N
lambda_val_2 = 0.13/N

# Définition des fenêtres
w_hann = np.hanning(N)
w_base = np.hanning(N) + 1e-4

# Passage à des enveloppes bilatérales symétriques par rapport au centre N//2
t_centered = np.abs(t - N//2)
w_exp_rapide = epsilon1 * np.exp(- lambda_val_1 * t_centered)
w_exp_lente = epsilon2 * np.exp(- lambda_val_2 * t_centered)

# La combinaison multiplicative reste identique
w_reg = w_hann * (w_exp_rapide + w_exp_lente)


# ---------------------------------------------------------
# AJOUT : Vérification de la contrainte Overlap-Add (OLA)
# ---------------------------------------------------------
n_hop = N//20  # Votre pas de recouvrement (ex: 256 pour un overlap de 75%)
total_length = 5 * N  # Longueur du signal de test
ola_buffer = np.zeros(total_length)

# Simulation du fenêtrage glissant (Somme des fenêtres au carré pour iSTFT)
for hop_step in range(0, total_length - N, n_hop):
    ola_buffer[hop_step:hop_step + N] += w_reg**2

# Isoler la zone centrale stable pour éliminer les effets de bord d'allumage
stable_zone = ola_buffer[2*N:3*N]
mean_ola = np.mean(stable_zone)
peak_to_peak_fluctuation = np.max(stable_zone) - np.min(stable_zone)
relative_error_db = 20 * np.log10(peak_to_peak_fluctuation / mean_ola + 1e-12)

print(f"Denominator peak-to-peak fluctuation: {peak_to_peak_fluctuation:.2e}")
print(f"Relative modulation error: {relative_error_db:.2f} dB")


# ---------------------------------------------------------
# Calcul des réponses fréquentielles
# ---------------------------------------------------------
N_fft = 32*N
W_hann = np.fft.fft(w_hann, N_fft)
W_base = np.fft.fft(w_base, N_fft)
W_reg = np.fft.fft(w_reg, N_fft)
freqs = np.fft.fftshift(np.fft.fftfreq(N_fft))
W_hann_mag = 20 * np.log10(np.abs(np.fft.fftshift(W_hann)) / np.max(np.abs(W_hann)) + 1e-12)
W_base_mag = 20 * np.log10(np.abs(np.fft.fftshift(W_base)) / np.max(np.abs(W_base)) + 1e-12)
W_reg_mag  = 20 * np.log10(np.abs(np.fft.fftshift(W_reg)) / np.max(np.abs(W_reg)) + 1e-12)

print(f"Maximum attenuation -------- Hann (no eps): {W_hann_mag[-1]:.0f} dB ; Hann + 1e-4: {W_base_mag[-1]:.0f} dB ; Regularized: {W_reg_mag[-1]:.0f} dB")


# ---------------------------------------------------------
# Tracé des graphiques
# ---------------------------------------------------------
plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20
})

# 1. Figure combinée (Domaine temporel + Réponse fréquentielle)
fig_comb, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Domaine temporel
# ax1.scatter(t[::10], w_hann[::10], label='Hann', color='green', marker="x", linewidths=2)

# ax1.plot(t, w_hann, label=r'Hann', color='green',linestyle='--', lw =8)
ax1.plot(t, w_reg, label='Proposed window', color='red', linestyle='--', lw=8)
ax1.set_xlabel('Samples', fontsize=18)
ax1.set_ylabel('Amplitude', labelpad=15, fontsize=18)

ax1.grid(False)
ax1.tick_params(labelsize=16)

# Subplot 2: Réponse fréquentielle
ax2.plot(freqs, W_hann_mag, label='Hann', color='green', linestyle='--', alpha=0.7)
# ax2.plot(freqs, W_base_mag, label=r'Hann + $\varepsilon$', color='blue', alpha=0.5)
ax2.plot(freqs, W_reg_mag, label='Proposed window', color='red', linestyle='--')
ax2.set_xlabel('Normalized Frequency', fontsize=18)
ax2.set_ylabel('Magnitude (dB)', labelpad=10, fontsize=18)
ax2.set_ylim([-180, max(np.max(W_hann_mag), np.max(W_base_mag), np.max(W_reg_mag)) + 10])
ax2.set_xlim(-0.25, 0.25)

ax2.grid(True)
ax2.tick_params(labelsize=16)

# Légende en bloc en bas au centre, 1 ligne, 3 colonnes, très grosse police
handles, labels = ax1.get_legend_handles_labels()
fig_comb.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=20, frameon=True)

# Ajustement des marges pour laisser de la place en bas pour la légende
fig_comb.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.20, wspace=0.22)

plt.savefig('/home/adubois/openunmix/OpenUnmix/fig/hann_window.png', bbox_inches='tight')
plt.close()