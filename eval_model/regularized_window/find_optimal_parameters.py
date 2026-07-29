import numpy as np
import matplotlib.pyplot as plt
import itertools

N = 683
t = np.arange(N)
n_hop = N // 20

# Definition de la fenetre de base et de l'axe temporel centre
w_hann = np.hanning(N)
t_centered = np.abs(t - N//2)

# Definition de la grille de recherche autour des valeurs initiales
epsilon1_vals = np.linspace(0.0005, 0.2, 20)
lambda_1_coeffs = np.linspace(0.5,5,50) # are divided by N
lambda_2_coeffs = np.linspace(0.001,1,40)

# Variables pour stocker le minimum
best_loss = float('inf')
best_params = None
best_w_reg = None

# Facteurs de normalisation pour la loss car les ordres de grandeur different
# L'erreur OLA est tres petite (~1e-4), la norme du gradient est grande (~100)
alpha_ola = 1e4
beta_grad = 1.0

for eps1, l1_c, l2_c in itertools.product(epsilon1_vals, lambda_1_coeffs, lambda_2_coeffs):
    eps2 = 1.0 - eps1
    l1 = l1_c / N
    l2 = l2_c / N
    
    # Construction de la fenetre
    w_exp_rapide = eps1 * np.exp(- l1 * t_centered)
    w_exp_lente = eps2 * np.exp(- l2 * t_centered)
    w_reg = w_hann * (w_exp_rapide + w_exp_lente)
    
    # 1. Calcul de la penalite Overlap-Add (Peak-to-Peak)
    total_length = 4 * N
    ola_buffer = np.zeros(total_length)
    for hop_step in range(0, total_length - N + 1, n_hop):
        ola_buffer[hop_step:hop_step + N] += w_reg**2
    
    stable_zone = ola_buffer[N:2*N]
    ola_ptp = np.max(stable_zone) - np.min(stable_zone)
    
    # 2. Calcul de la penalite sur la reponse frequentielle (Norme du gradient)
    N_fft = 8192
    W_reg = np.fft.fft(w_reg, N_fft)
    freqs = np.fft.fftfreq(N_fft)
    W_reg_mag = 20 * np.log10(np.abs(W_reg) + 1e-12)
    
    mask = freqs >= 0
    W_reg_mag_pos = W_reg_mag[mask]
    
    gradient = np.gradient(W_reg_mag_pos)
    grad_norm = np.linalg.norm(gradient)
    
    # Ajout d'une penalite specifique sur les remontees (oscillations strictes)
    # Les derivees positives indiquent une perte de monotonie
    oscillation_penalty = np.sum(np.maximum(0, gradient))
    
    # 3. Calcul de la Loss globale
    loss = alpha_ola * ola_ptp + beta_grad * (grad_norm + 2 * oscillation_penalty)
    
    if loss < best_loss:
        best_loss = loss
        best_params = (eps1, l1_c, l2_c)
        best_w_reg = w_reg

print("Recherche terminee.")
print(f"Meilleurs parametres trouves :")
print(f"epsilon1 = {best_params[0]}")
print(f"lambda_val_1 = {best_params[1]}/N")
print(f"lambda_val_2 = {best_params[2]}/N")
print(f"Loss minimale : {best_loss:.4f}")

# Visualisation de la meilleure configuration trouvee
N_fft = 8192
W_best = np.fft.fft(best_w_reg, N_fft)
freqs = np.fft.fftshift(np.fft.fftfreq(N_fft))
W_best_mag = 20 * np.log10(np.abs(np.fft.fftshift(W_best)) + 1e-12)

mask = freqs >= 0
freqs_pos = freqs[mask]
W_best_mag_pos = W_best_mag[mask]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t, best_w_reg, color='red')
ax1.set_title(f'Meilleure Fenetre Temporelle (eps1={best_params[0]}, l1={best_params[1]}/N, l2={best_params[2]}/N)')
ax1.set_xlabel('Echantillons')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

ax2.plot(freqs_pos, W_best_mag_pos, color='red')
ax2.set_title('Reponse Frequentielle (Module)')
ax2.set_xlabel('Frequence normalisee')
ax2.set_ylabel('Magnitude (dB)')
ax2.grid(True)

plt.tight_layout()
plt.show()