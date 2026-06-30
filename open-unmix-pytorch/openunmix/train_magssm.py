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

import data
import model
import utils
import transforms
import magssm_umx
from path_config import setup_paths, amp_autocast, amp_grad_scaler
setup_paths()

tqdm.monitor_interval = 0


# ---------------------------------------------------------------------------
# Tracking of SSM Lambda eigenvalues (similar to train_mask.py)
# ---------------------------------------------------------------------------

def collect_lambda_stats(model) -> dict:
    """
    Traverses the Progressive_SSM layers inside the model and collects
    statistics on Lambda * Delta.
    """
    try:
        from model_edge.ssm_bis import Progressive_SSM
    except ImportError:
        return {}

    stats = {}
    base = model.module if hasattr(model, 'module') else model

    for name, mod in base.named_modules():
        if not isinstance(mod, Progressive_SSM):
            continue

        with torch.no_grad():
            L = mod.Lambda.detach().float()           # [N, 2]
            log_step = mod.log_step.detach().float()  # [N]
            step = (mod.step_scale * torch.exp(log_step))  # Delta [N]

            Lambda_c = torch.complex(L[:, 0], L[:, 1])
            LD = Lambda_c * step
            ld_re = LD.real
            ld_im = LD.imag

            Lambda_bar = torch.exp(LD)
            mag = Lambda_bar.abs()

            n_nan = torch.isnan(L).any().item()

            stats[name] = {
                "ld_re_mean": float(ld_re.mean()) if not n_nan else None,
                "ld_re_min":  float(ld_re.min())  if not n_nan else None,
                "ld_re_max":  float(ld_re.max())  if not n_nan else None,
                "ld_im_mean": float(ld_im.mean()) if not n_nan else None,
                "ld_im_min":  float(ld_im.min())  if not n_nan else None,
                "ld_im_max":  float(ld_im.max())  if not n_nan else None,
                "lbar_mag_mean": float(mag.mean()) if not n_nan else None,
                "lbar_mag_max":  float(mag.max())  if not n_nan else None,
                "n_unstable":  int((ld_re > 0.0).sum())    if not n_nan else -1,
                "n_near_zero": int((ld_re > -1e-3).sum())  if not n_nan else -1,
                "n_nan_params": int(torch.isnan(L).sum()),
            }
    return stats


def train(args, unmix, encoder, device, train_sampler, optimizer, scaler=None):
    losses = utils.AverageMeter()
    nan_batches = 0
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        use_amp = (scaler is not None) and (device.type == "cuda")
        with amp_autocast(enabled=use_amp):
            Y_hat = unmix(x)
            Y = encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)

        if not torch.isfinite(loss):
            nan_batches += 1
            print(f"[WARN] Non-finite loss ({loss.item():.4g}) on this batch — batch ignored.")
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unmix.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unmix.parameters(), max_norm=1.0)
            optimizer.step()

        losses.update(loss.item(), Y.size(1))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))

    if nan_batches > 0:
        print(f"[WARN] {nan_batches} batch(es) ignored (NaN/Inf) during this epoch.")
    return losses.avg if losses.count > 0 else float('nan')


def valid(args, unmix, encoder, device, valid_sampler, use_amp=False):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)

            with amp_autocast(enabled=use_amp and (device.type == "cuda")):
                Y_hat = unmix(x)
                Y = encoder(y)
                loss = torch.nn.functional.mse_loss(Y_hat, Y)

            if not torch.isfinite(loss):
                continue
            losses.update(loss.item(), Y.size(1))
        return losses.avg if losses.count > 0 else float('nan')


def load_ssm_params(ssm_model_path):
    """Auto-detect parameters from ssm model configuration."""
    ssm_path = Path(ssm_model_path).expanduser()
    sep_json = ssm_path / "separator.json"
    conf = {}
    if sep_json.exists():
        with open(sep_json, "r") as f:
            conf = json.load(f)
    else:
        voc_json = ssm_path / "vocals.json"
        if voc_json.exists():
            with open(voc_json, "r") as f:
                vconf = json.load(f)
                conf = vconf.get("args", {})
                
    return {
        "nfft": conf.get("nfft", 682),
        "nhop": conf.get("nhop", 34),
        "nb_magssm_states": conf.get("nb_magssm_states", 682),
        "d_out": conf.get("d_out", None),
        "og": conf.get("og", False),
        "eps_stability": conf.get("eps_stability", 1e-3),
        "dt_min": conf.get("dt_min", 0.001),
        "dt_max": conf.get("dt_max", 0.1)
    }


def load_backbone_params(backbone_path):
    """Auto-detect parameters from backbone configuration."""
    path = Path(backbone_path).expanduser()
    voc_json = path / "vocals.json"
    if voc_json.exists():
        with open(voc_json, "r") as f:
            conf = json.load(f)
            args_dict = conf.get("args", {})
            return (
                args_dict.get("hidden_size", 512),
                args_dict.get("nb_layers", 3)
            )
    return 512, 3


def main():
    parser = argparse.ArgumentParser(description="MagSSM OpenUnmix Fine-tuner")

    # which target do we want to train?
    parser.add_argument("--target", type=str, default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset parameters
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
    parser.add_argument("--root", type=str, help="root path of dataset")
    parser.add_argument("--output",
        type=str,
        default="open-unmix-magssm",
        help="provide output path base folder name",
    )
    
    # Pre-trained models paths
    parser.add_argument("--model", type=str, required=True,
        help="Path of pre-trained OpenUnmix backbone checkpoint directory"
    )
    parser.add_argument("--ssm-model", type=str, required=True,
        help="Path of pre-trained trainable spectrogram SSM checkpoint directory"
    )
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to resume training")
    parser.add_argument("--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`)",
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--patience",
        type=int,
        default=140,
        help="maximum number of bad epochs (default: 140)",
    )
    parser.add_argument("--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument("--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed")

    # Freezing and architecture customization
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
        help="Freeze the OpenUnmix separation backbone and only optimize the SSM encoder parameters"
    )
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone",
        help="Do not freeze the separation backbone"
    )
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
        help="Learning rate for backbone parameters when not frozen (Scenario B)"
    )

    # Model parameters override (if they need override, otherwise auto-detected)
    parser.add_argument("--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds",
    )
    parser.add_argument("--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument("--chunk-dur", type=float, default=6.0,
        help="chunk duration in seconds for SSM computation"
    )
    parser.add_argument("--mel", action="store_true", default=False,
        help="SSM initialized with mel scale log frequency spacing"
    )
    parser.add_argument("--eps-stability", type=float, default=1e-3,
        help="SSM eps stability parameter"
    )
    parser.add_argument("--dt-min", type=float, default=0.001,
        help="SSM minimum time step"
    )
    parser.add_argument("--dt-max", type=float, default=0.1,
        help="SSM maximum time step"
    )
    parser.add_argument("--og", action="store_true", default=False,
        help="SSM original flag"
    )
    parser.add_argument("--bandwidth", 
                        type=int, default=16000, help="maximum model bandwidth in hz"
    )
    parser.add_argument("--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model",
    )
    parser.add_argument("--nb-workers", 
                        type=int, default=0, help="Number of workers for dataloader."
    )
    parser.add_argument("--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )
    parser.add_argument("--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument("--no-cuda", 
                        action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision (AMP) during training"
    )

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    try:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        repo = Repo(repo_dir)
        commit = repo.head.commit.hexsha[:7]
    except Exception:
        commit = "unknown"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    args.sample_rate = train_dataset.sample_rate

    # Create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # 1. Load configuration and weights of SSM model
    ssm_conf = load_ssm_params(args.ssm_model)
    nfft = ssm_conf["nfft"]
    nhop = ssm_conf["nhop"]
    nb_magssm_states = ssm_conf["nb_magssm_states"]
    d_out = ssm_conf["d_out"]
    d_out = nb_magssm_states if d_out is None else d_out
    
    og = ssm_conf["og"]
    eps_stability = ssm_conf["eps_stability"]
    dt_min = ssm_conf["dt_min"]
    dt_max = ssm_conf["dt_max"]
    
    print(f"[SSM Params] nfft={nfft}, nhop={nhop}, nb_states={nb_magssm_states}, d_out={d_out}")
    print(f"[SSM Stability] og={og}, eps_stability={eps_stability}, dt_min={dt_min}, dt_max={dt_max}")

    # 2. Detect backbone normalizer and layout params from pretrained OpenUnmix backbone
    backbone_dir = Path(args.model).expanduser()
    backbone_file = next(backbone_dir.glob("%s*.chkpnt" % args.target), None)
    if not backbone_file:
        backbone_file = next(backbone_dir.glob("%s*.pth" % args.target), None)
    if not backbone_file:
        raise FileNotFoundError(f"Could not find checkpoint for target {args.target} under {args.model}")

    print(f"Inspecting backbone weights from {backbone_file} to auto-detect configuration...")
    backbone_state = torch.load(backbone_file, map_location="cpu")
    state_dict_to_inspect = backbone_state.get("state_dict", backbone_state)

    has_ln = any("ln1" in k for k in state_dict_to_inspect.keys())
    hidden_size, nb_layers = load_backbone_params(args.model)
    print(f"[Backbone Auto-Detected] use_layernorm={has_ln}, hidden_size={hidden_size}, nb_layers={nb_layers}")

    # Set parameters to args for JSON writing
    args.use_layernorm = has_ln
    args.hidden_size = hidden_size
    args.nb_layers = nb_layers
    args.nb_magssm_states = nb_magssm_states
    args.d_out = d_out
    args.nfft = nfft
    args.nhop = nhop
    args.og = og
    args.eps_stability = eps_stability
    args.dt_min = dt_min
    args.dt_max = dt_max

    # Load input mean and scale from the pre-trained backbone if present
    scaler_mean = None
    scaler_std = None
    if "input_mean" in state_dict_to_inspect:
        scaler_mean = -state_dict_to_inspect["input_mean"].numpy()
    if "input_scale" in state_dict_to_inspect:
        scaler_std = 1.0 / state_dict_to_inspect["input_scale"].numpy()

    # STFT reference encoder
    stft, _ = transforms.make_filterbanks(
        n_fft=nfft, n_hop=nhop, sample_rate=train_dataset.sample_rate
    )
    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    chunk_duration_in_frames = int(args.chunk_dur * args.sample_rate)

    # Initialize the combined model
    unmix = magssm_umx.MagSSM_OpenUnmix(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_bins=nfft // 2 + 1,
        nb_channels=args.nb_channels,
        hidden_size=hidden_size,
        nb_layers=nb_layers,
        unidirectional=args.unidirectional,
        dim_state=nb_magssm_states,
        d_out=d_out,
        n_fft=nfft,
        n_hop=nhop,
        encoder=encoder,
        device=device,
        chunk_duration=chunk_duration_in_frames,
        log_distributed_frequencies=args.mel,
        use_layernorm=has_ln,
        og=og,
        eps_stability=eps_stability,
        dt_min=dt_min,
        dt_max=dt_max
    ).to(device)

    # Load SSM weights
    ssm_file = next(Path(args.ssm_model).expanduser().glob("%s*.chkpnt" % args.target), None)
    if not ssm_file:
        ssm_file = next(Path(args.ssm_model).expanduser().glob("%s*.pth" % args.target), None)
    
    if ssm_file:
        print(f"Loading pre-trained SSM weights from {ssm_file}")
        ssm_state = torch.load(ssm_file, map_location=device)
        ssm_state_dict = ssm_state.get("state_dict", ssm_state)
        # Filter keys corresponding to magssm_encoder
        ssm_filtered = {k: v for k, v in ssm_state_dict.items() if "magssm_encoder" in k}
        unmix.load_state_dict(ssm_filtered, strict=False)
    else:
        print("[WARN] Pre-trained SSM weights not found. Initializing randomly.")

    # Load Backbone weights
    print(f"Loading pre-trained separation backbone weights from {backbone_file}")
    backbone_filtered = {k: v for k, v in state_dict_to_inspect.items() if "magssm_encoder" not in k and "encoder" not in k}
    unmix.load_state_dict(backbone_filtered, strict=False)

    # Apply Freezing of Backbone
    if args.freeze_backbone:
        for name, param in unmix.named_parameters():
            if "magssm_encoder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("[INFO] Backbone weights FROZEN. Fine-tuning only the MagSSM encoder.")
    else:
        print("[INFO] Separation backbone weights are TRAINABLE.")

    # Setup Optimizer and Scheduler
    trainable_params = list(filter(lambda p: p.requires_grad, unmix.parameters()))
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total number of trainable parameters: {total_trainable}")

    if args.freeze_backbone:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Scenario B: co-optimization with differential learning rates
        ssm_params = []
        backbone_params = []
        for name, param in unmix.named_parameters():
            if "magssm_encoder" in name:
                ssm_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer_groups = [
            {"params": ssm_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr_backbone}
        ]
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
        print(f"[INFO] Scenario B: Co-optimization enabled.")
        print(f"       SSM Encoder learning rate: {args.lr}")
        print(f"       OpenUnmix Backbone learning rate: {args.lr_backbone}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)
    scaler = amp_grad_scaler(enabled=args.amp and use_cuda)

    # Handle checkpoint resume
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    lambda_log_path = Path(target_path, args.target + "_lambda.json")
    if lambda_log_path.exists():
        with open(lambda_log_path) as f:
            lambda_history = json.load(f)
    else:
        lambda_history = []

    # Write configuration details
    separator_conf = {
        "nfft": nfft,
        "nhop": nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
        "nb_magssm_states": nb_magssm_states,
        "use_layernorm": has_ln,
        "backbone_frozen": args.freeze_backbone,
    }
    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        
        train_loss = train(args, unmix, encoder, device, train_sampler, optimizer, scaler=scaler)
        valid_loss = valid(args, unmix, encoder, device, valid_sampler, use_amp=args.amp)
        scheduler.step(valid_loss)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)
        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target,
        )

        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            "commit": commit,
        }

        with open(Path(target_path, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        # Eigenvalues trajectory logging
        lambda_stats = collect_lambda_stats(unmix)
        lambda_history.append({"epoch": epoch, "lambda": lambda_stats})
        with open(lambda_log_path, "w") as f:
            json.dump(lambda_history, f, indent=2)

        for mod_name, s in lambda_stats.items():
            if s.get("n_nan_params", 0) > 0:
                print(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: NaN in Lambda ({s['n_nan_params']} params)")
            elif s.get("n_unstable", 0) > 0:
                print(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: {s['n_unstable']} unstable states (Re(Λ·Δ)>0), ld_re_max={s['ld_re_max']:.3e}")
            elif s.get("ld_re_max") is not None and s["ld_re_max"] > -1e-3:
                print(f"[Lambda] ! EPOCH {epoch} — {mod_name}: ld_re_max={s['ld_re_max']:.3e} (Re(Λ·Δ) close to 0)")

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
