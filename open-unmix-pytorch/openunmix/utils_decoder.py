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

import decoder
from transforms import make_filterbanks, ComplexNorm


def load_target_models(targets, model_str_or_path="umxl", device="cpu", pretrained=True):
    """Core model loader for Trainable_decoder

    target model path can be either <target>.pth, or <target>.chkpnt
    """
    if isinstance(targets, str):
        targets = [targets]

    model_path = Path(model_str_or_path).expanduser()
    if not model_path.exists():
        # model path does not exist, try hub loader if available
        try:
            hub_loader = getattr(openunmix, model_str_or_path + "_decoder")
            err = io.StringIO()
            with redirect_stderr(err):
                return hub_loader(targets=targets, device=device, pretrained=pretrained)
            print(err.getvalue())
        except AttributeError:
            raise NameError("Model does not exist on torchhub")
    else:
        models = {}
        for target in targets:
            # load model config from disk
            with open(Path(model_path, target + ".json"), "r") as stream:
                results = json.load(stream)

            ds = results["args"].get("ds", 1)
            sample_rate = results["args"]["sample_rate"] // ds
            seq_dur = results["args"].get("seq_dur", 6.0)
            
            length = results["args"].get("length", None)
            if length is None:
                length = int(seq_dur * sample_rate)

            chunk_dur = results["args"].get("chunk_dur", 6.0)
            chunk_duration_in_frames = int(chunk_dur * sample_rate)

            target_model_path = next(Path(model_path).glob("%s*.chkpnt" % target), None)
            if target_model_path is None:
                target_model_path = next(Path(model_path).glob("%s*.pth" % target))
                
            checkpoint = torch.load(target_model_path, map_location=device)
            state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

            models[target] = decoder.Trainable_decoder(
                n_fft=results["args"]["nfft"],
                n_hop=results["args"]["nhop"],
                length=length,
                dim_state=results["args"]["nb_magssm_states"],
                og=results["args"].get("og", False),
                B_C_init="orthogonal",
                device=device,
                chunk_duration=chunk_duration_in_frames,
                log_distributed_frequencies=results["args"].get("mel", False),
                eps_stability=results["args"].get("eps_stability", 1e-3),
                dt_min=results["args"].get("dt_min", 0.001),
                dt_max=results["args"].get("dt_max", 0.1),
                re_lower=results["args"].get("re_lower", None),
                re_upper=results["args"].get("re_upper", None),
                ensure_stability=results["args"].get("ensure_stability", "abs"),
                sigmoid_scale=results["args"].get("sigmoid_scale", 1.0),
                structured_initialisation=results["args"].get("structured_initialisation", False)
            )

            if pretrained:
                models[target].load_state_dict(state, strict=False)

            models[target].to(device)
        return models