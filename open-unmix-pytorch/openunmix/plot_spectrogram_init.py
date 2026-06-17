from spectrogram import Trainable_spectrogram
import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import torchaudio
import shutil


import data
import model
import utils
import transforms
import sedge_mask
from spectrogram import Trainable_spectrogram
import utils_spectrogram
from path_config import amp_autocast, amp_grad_scaler
import pylab 

import matplotlib.pyplot as plt





device="cpu"

downsampling = 3
nfft= 256
nhop = nfft//4
nbins = 129
nstates = nbins 
chunk_duration = 2.0
sample_rate = 44100 // downsampling
chunk_duration_in_frames = int(sample_rate * chunk_duration)
eps_stability = 0
mel = True

def plot_spectrogram_from_array(
    spectrogram: np.ndarray,
    power_exp: float = 0.3,
    sample_rate: int = 44100,
    nhop: int = 1024,
    freq_max: float = sample_rate//2,
    name: str = "Spectrogram",
    ax: plt.Axes = None,
) -> plt.Axes:

    n_freq, n_frames = spectrogram.shape

    freqs = np.linspace(0, n_freq) 

    times = np.arange(n_frames) * nhop / sample_rate  

    display = np.abs(spectrogram) ** power_exp

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))

    pcm = ax.pcolormesh(times, freqs, display, shading="nearest", cmap="inferno")
    plt.colorbar(pcm, ax=ax, label=f"Magnitude^{power_exp:.2f}")

    ax.set_title(
        f"{name}  |  power_exp={power_exp:.2f}  |  nhop={nhop}",
        fontsize=12,
    )
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [s]")
    ax.set_ylim(0, freq_max)
    ax.set_xlim(times[0], times[-1])

    return ax

def plot_single(
    spectrograms: np.ndarray,
    channel: int = 0,
    power_exp: float = 0.3,
    sample_rate: int = sample_rate,
    nhop: int = nhop,
    freq_max: float = sample_rate//2,
    name: str = "",
    save_fig: str = "/home/adubois/openunmix/OpenUnmix/fig/spectrogram_init.png",
):


    """Affiche un seul spectrogramme (true ou predicted)."""

    sp = spectrograms[0,channel,:,:]  # (F, T)

    fig, ax = plt.subplots(figsize=(15, 5))
    plot_spectrogram_from_array(
        sp,
        power_exp=power_exp,
        sample_rate=sample_rate,
        nhop=nhop,
        freq_max=freq_max,
        name=f"Trainable spectrogram at initialisation : {nbins} bins , {nstates} states — channel {channel}  |  {name}",
        # name=f"True spectrogram at initialisation : {nbins} bins  — channel {channel}  |  {name}",

        ax=ax,
    )
    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_fig}")
    else:
        plt.show()





def main():
    parser = argparse.ArgumentParser(description="Open trainable_spectrogram Trainer")

    parser.add_argument("--target", type=str, default="vocals",
        help="target source (will be passed to the dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")


    parser.add_argument("--root", type=str, help="root path of dataset")

    parser.add_argument("--seq-dur",
        type=float,
        default=30,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )

    args, _ = parser.parse_known_args()
    


    _, valid_dataset, _ = data.load_datasets(parser, args)
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **{})



    stft, _ = transforms.make_filterbanks(
            n_fft=nfft, n_hop=nhop, sample_rate=sample_rate
        )


    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=False)).to(device)

    spectrogram = Trainable_spectrogram(
        nb_bins = nbins,
        n_hop = nfft//4,
        dim_state = nstates,
        C_C_init= "diagonal",
        encoder = encoder,
        device = device,
        chunk_duration=chunk_duration_in_frames,
        log_distributed_frequencies = mel, 
        eps_stability = eps_stability
    )

    spectrogram.eval()
    with torch.no_grad():
        for x, _ in valid_sampler:
            x = x.to(device)
            x = x[:, : , ::downsampling]

            X_hat = spectrogram(x).detach()
            X = encoder(x).detach()
            
         
            X_hat = X_hat[0,0,:,0:5000]
            
            plt.imshow(X_hat**0.3, aspect="auto", origin = "lower")

            plt.savefig("/home/adubois/openunmix/OpenUnmix/fig/spectrogram_init.png")

            

            break






if __name__ == "__main__":
    main()
