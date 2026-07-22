import argparse
import os
import torch
import torch.distributed as dist

# Set CUDA device ASAP, before importing anything else (like torchaudio, which might initialize CUDA)
is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if is_distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
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


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tracking of SSM Lambda eigenvalues (similar to train_mask.py)
# ---------------------------------------------------------------------------

def collect_lambda_stats(model) -> dict:
    """
    Traverses the Progressive_SSM layers inside the model and collects
    statistics on Lambda * Delta.
    """
    try:
        import model_edge
    except ImportError:
        return {}

    stats = {}
    # If DDP wrapped, get underlying module
    base = model.module if hasattr(model, 'module') else model

    for name, mod in base.named_modules():
        if mod.__class__.__name__ not in ["SSM", "Progressive_SSM"]:
            continue

        with torch.no_grad():
            L = mod.Lambda.detach().float()           # [N, 2]
            log_step = mod.log_step.detach().float()  # [N]
            step = (mod.step_scale * torch.exp(log_step))  # Delta [N]

            # Calcul de la partie réelle effective selon le mode de stabilité du forward
            if mod.ensure_stability == 'sigmoid_interval':
                # produit effectif = Re(Lambda_c) * Delta
                width = mod.re_upper - mod.re_lower
                ld_re = mod.re_lower + width * torch.sigmoid(mod.sigmoid_scale * L[:, 0])
            else:
                ld_re = L[:, 0] * step  # taux d'amortissement effectif standard

            ld_im = L[:, 1] * step  # fréquence en rad/sample
            LD = torch.complex(ld_re, ld_im)

            Lambda_bar = torch.exp(LD)
            mag = Lambda_bar.abs()

            n_nan = torch.isnan(L).any().item()

            alpha = getattr(model, 'alpha', getattr(base, 'alpha', 0.0))
            beta = getattr(model, 'beta', getattr(base, 'beta', 0.0))

            excess_pos = torch.clamp(ld_im - torch.pi, min=0.0)
            w_pos = excess_pos + beta * (excess_pos > 0).float()

            excess_neg = torch.clamp(-ld_im, min=0.0)
            w_neg = excess_neg + beta * (excess_neg > 0).float()

            loss_regularization = alpha * (w_pos.pow(2) + w_neg.pow(2)).mean()

            stats[name] = {
                "ld_re_mean": float(ld_re.mean()) if not n_nan else None,
                "ld_re_min":  float(ld_re.min())  if not n_nan else None,
                "ld_re_max":  float(ld_re.max())  if not n_nan else None,
                "ld_im_mean": float(ld_im.mean()) if not n_nan else None,
                "ld_im_min":  float(ld_im.min())  if not n_nan else None,
                "ld_im_max":  float(ld_im.max())  if not n_nan else None,
                "lbar_mag_mean": float(mag.mean()) if not n_nan else None,
                "lbar_mag_max":  float(mag.max())  if not n_nan else None,
                # Terme de régularisation L2_im_lambda pour ce module
                "loss_regularization": float(loss_regularization.item()) if not n_nan else None,
                "n_unstable":  int((ld_re > 0.0).sum())    if not n_nan else -1,
                "n_near_zero": int((ld_re > -1e-3).sum())  if not n_nan else -1,
                "n_nan_params": int(torch.isnan(L).sum()),
            }
    return stats


def train(args, unmix, encoder, device, train_sampler, optimizer, is_distributed=False, scaler=None):
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
            if not args.quiet:
                print_rank0(f"[WARN] Non-finite loss ({loss.item():.4g}) on this batch — batch ignored.")
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
        print_rank0(f"[WARN] {nan_batches} batch(es) ignored (NaN/Inf) during this epoch.")

    if is_distributed:
        loss_sum = torch.tensor(losses.sum if losses.count > 0 else 0.0, device=device)
        count_sum = torch.tensor(float(losses.count), device=device)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)
        if count_sum.item() == 0:
            return float('nan')
        return (loss_sum / count_sum).item()
    else:
        return losses.avg if losses.count > 0 else float('nan')


def valid(args, unmix, encoder, device, valid_sampler, is_distributed=False, use_amp=False):
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

    if is_distributed:
        loss_sum = torch.tensor(losses.sum if losses.count > 0 else 0.0, device=device)
        count_sum = torch.tensor(float(losses.count), device=device)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / count_sum).item() if count_sum.item() > 0 else float('nan')
    else:
        avg_loss = losses.avg if losses.count > 0 else float('nan')

    return avg_loss


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
    parser = argparse.ArgumentParser(description="Distributed MagSSM OpenUnmix Fine-tuner")

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
    parser.add_argument("--eps-stability", type=float, default=0,
        help="SSM eps stability parameter"
    )
    parser.add_argument("--dt-min", type=float, default=0.001,
        help="SSM minimum time step"
    )
    parser.add_argument("--dt-max", type=float, default=0.1,
        help="SSM maximum time step"
    )
    parser.add_argument("--og", action="store_true", default=True,
        help="SSM original flag"
    )
    parser.add_argument("--ensure-stability", type=str, default="abs",
        choices=["abs", "relu", "sigmoid_interval"],
        help="Stability enforcement mode for eigenvalue real parts."
    )
    parser.add_argument("--re-lower", type=float, default=None,
        help="Lower bound for Re(Lambda)*step product in sigmoid_interval mode."
    )
    parser.add_argument("--re-upper", type=float, default=None,
        help="Upper bound for Re(Lambda)*step product in sigmoid_interval mode."
    )
    parser.add_argument("--sigmoid-scale", type=float, default=1.0,
        help="Internal scaling factor inside the sigmoid for sigmoid_interval reparametrisation."
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

    # Distributed parameters
    parser.add_argument("--dist-backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use (default: nccl)")
    parser.add_argument("--master-addr", type=str, default=None,
                        help="IP address of the master node (rank 0).")
    parser.add_argument("--master-port", type=str, default="29500",
                        help="TCP port of the master node.")

    args, _ = parser.parse_known_args()

    # ---------------------------------------------------------------------------
    # Initialisation of distributed process group (DDP)
    # ---------------------------------------------------------------------------
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        if args.master_addr is not None:
            os.environ["MASTER_ADDR"] = args.master_addr
        if "MASTER_ADDR" not in os.environ:
            raise RuntimeError("MASTER_ADDR not defined. Use --master-addr or MASTER_ADDR environment variable.")
        os.environ.setdefault("MASTER_PORT", args.master_port)

        global_rank = int(os.environ["RANK"])
        local_rank  = int(os.environ["LOCAL_RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        global_rank = 0
        local_rank  = 0
        world_size  = 1
        device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    args.quiet = args.quiet or (global_rank != 0)

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print_rank0("Using GPU:", use_cuda)
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

    # ---------------------------------------------------------------------------
    # Loading dataset (barrier avoids concurrent folder writes/race conditions)
    # ---------------------------------------------------------------------------
    if is_distributed:
        if global_rank == 0:
            train_dataset, valid_dataset, args = data.load_datasets(parser, args)
        dist.barrier()
        if global_rank != 0:
            train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    else:
        train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    args.sample_rate = train_dataset.sample_rate

    # Create output dir (rank 0 only)
    target_path = Path(args.output)
    if global_rank == 0:
        target_path.mkdir(parents=True, exist_ok=True)

    # Distributed DDP Samplers
    if is_distributed:
        train_sampler_ddp = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True
        )
        valid_sampler_ddp = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False
        )
        train_sampler = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler_ddp, **dataloader_kwargs
        )
        valid_sampler = torch.utils.data.DataLoader(
            valid_dataset, batch_size=1, sampler=valid_sampler_ddp, **dataloader_kwargs
        )
    else:
        train_sampler_ddp = None
        train_sampler = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
        )
        valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # Load configuration and weights of SSM model
    ssm_conf = load_ssm_params(args.ssm_model)
    nfft = ssm_conf["nfft"]
    nhop = ssm_conf["nhop"]
    nb_magssm_states = ssm_conf["nb_magssm_states"]
    d_out = ssm_conf["d_out"]
    d_out = (nfft // 2 + 1) if d_out is None else d_out
    
    og = ssm_conf["og"]
    eps_stability = ssm_conf["eps_stability"]
    dt_min = ssm_conf["dt_min"]
    dt_max = ssm_conf["dt_max"]
    
    print_rank0(f"[SSM Params] nfft={nfft}, nhop={nhop}, nb_states={nb_magssm_states}, d_out={d_out}")
    print_rank0(f"[SSM Stability] og={og}, eps_stability={eps_stability}, dt_min={dt_min}, dt_max={dt_max}")

    # Detect backbone normalizer and layout params from pretrained OpenUnmix backbone
    backbone_dir = Path(args.model).expanduser()
    backbone_file = next(backbone_dir.glob("%s*.chkpnt" % args.target), None)
    if not backbone_file:
        backbone_file = next(backbone_dir.glob("%s*.pth" % args.target), None)
    if not backbone_file:
        raise FileNotFoundError(f"Could not find checkpoint for target {args.target} under {args.model}")

    print_rank0(f"Inspecting backbone weights from {backbone_file} to auto-detect configuration...")
    backbone_state = torch.load(backbone_file, map_location="cpu")
    state_dict_to_inspect = backbone_state.get("state_dict", backbone_state)

    has_ln = any("ln1" in k for k in state_dict_to_inspect.keys())
    hidden_size, nb_layers = load_backbone_params(args.model)
    print_rank0(f"[Backbone Auto-Detected] use_layernorm={has_ln}, hidden_size={hidden_size}, nb_layers={nb_layers}")

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

    # Initialize combined model
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
        dt_max=dt_max,
        re_lower=args.re_lower,
        re_upper=args.re_upper,
        ensure_stability=args.ensure_stability,
        sigmoid_scale=args.sigmoid_scale
    ).to(device)

    # Load SSM weights
    ssm_file = next(Path(args.ssm_model).expanduser().glob("%s*.chkpnt" % args.target), None)
    if not ssm_file:
        ssm_file = next(Path(args.ssm_model).expanduser().glob("%s*.pth" % args.target), None)
    
    if ssm_file:
        print_rank0(f"Loading pre-trained SSM weights from {ssm_file}")
        ssm_state = torch.load(ssm_file, map_location=device)
        ssm_state_dict = ssm_state.get("state_dict", ssm_state)
        # Filter keys corresponding to magssm_encoder
        ssm_filtered = {k: v for k, v in ssm_state_dict.items() if "magssm_encoder" in k}

        # --- On-the-fly conversion of Lambda weights to sigmoid_interval logits ---
        if args.ensure_stability == 'sigmoid_interval':
            lambda_key = "magssm_encoder.mimo.seq.Lambda"
            step_key = "magssm_encoder.mimo.seq.log_step"
            re_lower_key = "magssm_encoder.mimo.seq.re_lower"
            re_upper_key = "magssm_encoder.mimo.seq.re_upper"
            
            if lambda_key in ssm_filtered and step_key in ssm_filtered:
                L = ssm_filtered[lambda_key].clone()
                log_step = ssm_filtered[step_key]
                step_scale = unmix.magssm_encoder.mimo.seq.step_scale
                step = step_scale * torch.exp(log_step)
                
                # Check if the loaded weights are already in sigmoid_interval format
                was_sigmoid = False
                old_re_lower = None
                old_re_upper = None
                old_sigmoid_scale = 1.0
                
                # Try to load old config json first
                ssm_json = Path(args.ssm_model).expanduser() / f"{args.target}.json"
                if ssm_json.exists():
                    try:
                        with open(ssm_json, 'r') as f:
                            ssm_conf = json.load(f)
                        old_args = ssm_conf.get('args', {})
                        if old_args.get('ensure_stability') == 'sigmoid_interval':
                            was_sigmoid = True
                            old_re_lower = old_args.get('re_lower')
                            old_re_upper = old_args.get('re_upper')
                            old_sigmoid_scale = old_args.get('sigmoid_scale', 1.0)
                    except Exception as e:
                        print_rank0(f"[WARN] Error reading pre-trained SSM json: {e}")
                
                # Fallback to checking keys in checkpoint
                if not was_sigmoid and re_lower_key in ssm_filtered and re_upper_key in ssm_filtered:
                    was_sigmoid = True
                    old_re_lower = ssm_filtered[re_lower_key].item()
                    old_re_upper = ssm_filtered[re_upper_key].item()
                
                if was_sigmoid:
                    # The loaded Lambda[:, 0] are already logits.
                    # If bounds and scaling factors match, load directly; otherwise convert logits -> physical -> new logits.
                    if (old_re_lower == args.re_lower and 
                        old_re_upper == args.re_upper and 
                        old_sigmoid_scale == args.sigmoid_scale):
                        print_rank0("[INFO] Pre-trained weights are already in sigmoid_interval format with matching bounds. Loading directly.")
                    else:
                        print_rank0(f"[INFO] Pre-trained weights are in sigmoid_interval format but bounds mismatch (Old: [{old_re_lower}, {old_re_upper}], New: [{args.re_lower}, {args.re_upper}]). Converting logits...")
                        # 1. Reconstruct old physical product: Re(Lambda) * step
                        width_old = old_re_upper - old_re_lower
                        physical_product = old_re_lower + width_old * torch.sigmoid(old_sigmoid_scale * L[:, 0])
                        
                        # 2. Map to new bounds
                        width_new = args.re_upper - args.re_lower
                        physical_product = physical_product.clamp(min=args.re_lower, max=args.re_upper)
                        t = (physical_product - args.re_lower) / width_new
                        t = t.clamp(1e-6, 1.0 - 1e-6)
                        L[:, 0] = torch.log(t / (1.0 - t)) / args.sigmoid_scale
                        ssm_filtered[lambda_key] = L
                        print_rank0("[INFO] Lambda weights successfully converted to new bounds.")
                else:
                    # Loaded Lambda[:, 0] are standard real eigenvalues (e.g. from 'abs' or 'relu' stability modes)
                    print_rank0("[INFO] Converting pre-trained physical Lambda real parts to sigmoid_interval logits...")
                    # Physical product = Re(Lambda_c) * step
                    initial_product = L[:, 0] * step
                    
                    # Clamp within the target bounds
                    initial_product = initial_product.clamp(min=args.re_lower, max=args.re_upper)
                    
                    # Inverse sigmoid mapping
                    width_new = args.re_upper - args.re_lower
                    t = (initial_product - args.re_lower) / width_new
                    t = t.clamp(1e-6, 1.0 - 1e-6)
                    L[:, 0] = torch.log(t / (1.0 - t)) / args.sigmoid_scale
                    ssm_filtered[lambda_key] = L
                    print_rank0("[INFO] Lambda weights successfully converted.")

        # Remove old buffer values for re_lower/re_upper from checkpoint
        # so they don't overwrite the model's new bounds during load_state_dict
        for buf_key in ["magssm_encoder.mimo.seq.re_lower", "magssm_encoder.mimo.seq.re_upper"]:
            if buf_key in ssm_filtered:
                print_rank0(f"[INFO] Removing old buffer '{buf_key}' (value={ssm_filtered[buf_key].item():.6g}) from checkpoint to preserve new bounds.")
                del ssm_filtered[buf_key]

        unmix.load_state_dict(ssm_filtered, strict=False)
    else:
        print_rank0("[WARN] Pre-trained SSM weights not found. Initializing randomly.")

    # Load Backbone weights
    print_rank0(f"Loading pre-trained separation backbone weights from {backbone_file}")
    backbone_filtered = {k: v for k, v in state_dict_to_inspect.items() if "magssm_encoder" not in k and "encoder" not in k}
    unmix.load_state_dict(backbone_filtered, strict=False)

    # Apply Freezing of Backbone
    if args.freeze_backbone:
        for name, param in unmix.named_parameters():
            if "magssm_encoder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print_rank0("[INFO] Backbone weights FROZEN. Fine-tuning only the MagSSM encoder.")
    else:
        print_rank0("[INFO] Separation backbone weights are TRAINABLE.")

    # Setup Optimizer and Scheduler
    trainable_params = list(filter(lambda p: p.requires_grad, unmix.parameters()))
    total_trainable = sum(p.numel() for p in trainable_params)
    print_rank0(f"Total number of trainable parameters: {total_trainable}")

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
        print_rank0(f"[INFO] Scenario B: Co-optimization enabled.")
        print_rank0(f"       SSM Encoder learning rate: {args.lr}")
        print_rank0(f"       OpenUnmix Backbone learning rate: {args.lr_backbone}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)
    scaler = amp_grad_scaler(enabled=args.amp and use_cuda)

    # Resume from checkpoint (before DDP wrapping)
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

    # Wrap model in DDP (after checkpoint loading)
    if is_distributed:
        unmix = torch.nn.parallel.DistributedDataParallel(
            unmix,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    lambda_log_path = Path(target_path, args.target + "_lambda.json")
    if lambda_log_path.exists():
        with open(lambda_log_path) as f:
            lambda_history = json.load(f)
    else:
        lambda_history = []

    # Write configuration details (rank 0 only)
    if global_rank == 0:
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
        if is_distributed:
            train_sampler_ddp.set_epoch(epoch)

        t.set_description("Training epoch")
        end = time.time()
        
        train_loss = train(args, unmix, encoder, device, train_sampler, optimizer, is_distributed=is_distributed, scaler=scaler)
        valid_loss = valid(args, unmix, encoder, device, valid_sampler, is_distributed=is_distributed, use_amp=args.amp)
        scheduler.step(valid_loss)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)
        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        # Save checkpoint (rank 0 only)
        if global_rank == 0:
            raw_state_dict = unmix.module.state_dict() if is_distributed else unmix.state_dict()
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": raw_state_dict,
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
            raw_model = unmix.module if is_distributed else unmix
            lambda_stats = collect_lambda_stats(raw_model)
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
            print_rank0("Apply Early Stopping")
            break

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
