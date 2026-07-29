import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_regularized_window(N, epsilon1, lambda_coeff_1, lambda_coeff_2):
    t = np.arange(N)
    w_hann = np.hanning(N)
    
    epsilon2 = 1.0 - epsilon1
    lambda_val_1 = lambda_coeff_1 / N
    lambda_val_2 = lambda_coeff_2 / N
    
    t_centered = np.abs(t - N // 2)
    w_exp_rapide = epsilon1 * np.exp(-lambda_val_1 * t_centered)
    w_exp_lente = epsilon2 * np.exp(-lambda_val_2 * t_centered)
    
    w_reg = w_hann * (w_exp_rapide + w_exp_lente)
    return w_reg

def main():
    N = 683
    config_path = "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/configs.txt"
    fig_dir = "/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fig"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Parse configs.txt
    configs = []
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split at comment
                parts = line.split("#")
                param_part = parts[0].strip()
                comment_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Remove quotes
                param_part = param_part.replace('"', '').replace("'", "").strip()
                if not param_part:
                    continue
                
                tokens = param_part.split()
                if len(tokens) == 3:
                    try:
                        eps1 = float(tokens[0])
                        l1 = float(tokens[1])
                        l2 = float(tokens[2])
                        
                        # Clean comment part (remove indicators like 'x')
                        comment_clean = comment_part.split("x")[0].strip()
                        configs.append({
                            "eps1": eps1,
                            "l1": l1,
                            "l2": l2,
                            "label": comment_clean if comment_clean else f"eps1={eps1}, l1={l1}, l2={l2}"
                        })
                    except ValueError:
                        continue
    else:
        print(f"Warning: {config_path} not found. Using default configs.")
        # Fallback configs in case the file doesn't load
        configs = [
            {"eps1": 0.5000, "l1": 15.0000, "l2": 4.0000, "label": "-19.5dB"},
            {"eps1": 0.1000, "l1": 4.0000, "l2": 1.7152, "label": "-40dB"},
            {"eps1": 0.2000, "l1": 2.1694, "l2": 1.0000, "label": "-45dB"},
        ]






    plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20
})

    # Combined Plot: Time and Frequency domains
    fig_comb, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    t = np.arange(N)
    
    # Plot standard Hann window for reference
    w_hann = np.hanning(N)
    ax1.plot(t, w_hann, label="Hann (Base)", color="black", linestyle="--", linewidth=2.0)
    
    # Use a colormap to color the different configurations nicely
    try:
        cmap = plt.colormaps["turbo"].resampled(len(configs))
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("turbo", len(configs))
    
    for idx, cfg in enumerate(configs[::]):
        w_reg = get_regularized_window(N, cfg["eps1"], cfg["l1"], cfg["l2"])
        ax1.plot(t, w_reg, label=f"{cfg['label']} ", 
                 color=cmap(idx), alpha=0.85, linewidth=1.5)
                 
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: Frequency domain plot
    N_fft = 32768 # Very high resolution FFT
    freq = np.fft.rfftfreq(N_fft, d=1.0) # 0 to 0.5 cycles/sample
    
    # Plot standard Hann window frequency response
    W_hann = np.fft.rfft(w_hann, n=N_fft)
    W_hann_db = 20 * np.log10(np.abs(W_hann) / np.max(np.abs(W_hann)) + 1e-12)
    ax2.plot(freq, W_hann_db, label="Hann (Base)", color="black", linestyle="--", linewidth=2.0)
    
    for idx, cfg in enumerate(configs[::]):
        w_reg = get_regularized_window(N, cfg["eps1"], cfg["l1"], cfg["l2"])
        W_reg = np.fft.rfft(w_reg, n=N_fft)
        W_reg_db = 20 * np.log10(np.abs(W_reg) / np.max(np.abs(W_reg)) + 1e-12)
        ax2.plot(freq, W_reg_db, label=f"{cfg['label']} ", 
                 color=cmap(idx), alpha=0.85, linewidth=1.5)
                 
    ax2.set_xlabel("Normalized Frequency")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim(-120, 5) # Focus on relevant range
    ax2.set_xlim(0, 0.05) # Focus on full frequency range from 0 to 0.5
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Single legend box for the entire figure
    handles, labels = ax1.get_legend_handles_labels()
    fig_comb.legend(handles, labels, loc='lower center', ncol=6, fontsize=20, frameon=True)
    
    plt.tight_layout()
    # Leave room at the bottom for the legend
    plt.subplots_adjust(bottom=0.22)
    
    combined_plot_path = os.path.join(fig_dir, "w_reg_combined.png")
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {combined_plot_path}")

if __name__ == "__main__":
    main()
