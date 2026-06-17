from typing import Optional, Union

import torch
import torch.nn as nn
import os
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io
import json

import spectrogram 
from transforms import make_filterbanks, ComplexNorm


def load_target_models(targets, model_str_or_path="umxl", device="cpu", pretrained=True):
    """Core model loader

    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)

    The loader either loads the models from a known model string
    as registered in the __init__.py or loads from custom configs.
    """
    if isinstance(targets, str):
        targets = [targets]

    model_path = Path(model_str_or_path).expanduser()
    if not model_path.exists():
        # model path does not exist, use pretrained models
        try:
            # disable progress bar
            hub_loader = getattr(openunmix, model_str_or_path + "_spec")
            err = io.StringIO()
            with redirect_stderr(err):
                return hub_loader(targets=targets, device=device, pretrained=pretrained)
            print(err.getvalue())
        except AttributeError:
            raise NameError("Model does not exist on torchhub")
            # assume model is a path to a local model_str_or_path directory
    else:
        models = {}
        for target in targets:
            # load model from disk
            with open(Path(model_path, target + ".json"), "r") as stream:
                results = json.load(stream)

            stft, _ = make_filterbanks(
                n_fft=results["args"]["nfft"],
                n_hop = results["args"]["nhop"],
                sample_rate=results["args"]["sample_rate"])
                
            encoder = torch.nn.Sequential(
                stft, 
                ComplexNorm(mono=results["args"]["nb_channels"] == 1)
            ).to(device)    

            target_model_path = next(Path(model_path).glob("%s*.pth" % target))
            state = torch.load(target_model_path, map_location=device)

            models[target] = spectrogram.Trainable_spectrogram(
                nb_bins = results["args"]["nfft"] // 2 + 1,
                nb_channels = results["args"]["nb_channels"],
                n_hop = results["args"]["nhop"],
                dim_state = results["args"]["nb_magssm_states"],
                encoder = encoder,
                device = device,
                chunk_duration = int(results["args"]["chunk_dur"] * results["args"]["sample_rate"] / results["args"]["ds"]),
                log_distributed_frequencies = results["args"]["mel"]
            )

            if pretrained:
                models[target].load_state_dict(state, strict=False)

            models[target].to(device)
        return models