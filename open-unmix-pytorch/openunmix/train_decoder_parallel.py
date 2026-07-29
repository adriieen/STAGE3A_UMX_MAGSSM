import argparse
import os
import torch
import torch.distributed as dist

# Set CUDA device ASAP, before importing anything else
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
import shutil

import data
import model
import utils
import transforms
import sedge_mask
from decoder import Trainable_decoder
import utils_decoder
from path_config import amp_autocast, amp_grad_scaler
import pylab

tqdm.monitor_interval = 0


def save_spectrogram(X, save_path: str, complex_spectrogram=True):
    """
    Saves the spectrogram of a wavefile of format 1=batch, C=2, F, T in a .png file.
    Handles 5D complex tensors (B, C, F, T, 2) automatically.
    """
    if X.ndim == 5:
        X_re, X_im = X[..., 0], X[..., 1]
        X = X_re + 1j * X_im
    elif complex_spectrogram and torch.is_complex(X):
        pass
    else:
        raise Exception("spectrogram not in right format")

    X = torch.mean(X, dim=1)
    X = torch.abs(X[0])
    X = torch.log(X + 1e-8)
    pylab.imshow(X.detach().cpu().numpy(), aspect='auto', origin='lower')
    pylab.tight_layout()
    pylab.savefig(save_path, dpi=300)
    pylab.close()


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tracking des valeurs propres Lambda du SSM
# ---------------------------------------------------------------------------

def collect_lambda_stats(model) -> dict:
    """
    Parcourt tous les modules Progressive_SSM du modèle et collecte,
    pour chaque module, les statistiques du produit Lambda * Delta.
    """
    try:
        import model_edge
    except ImportError:
        return {}

    stats = {}
    base = model.module if hasattr(model, 'module') else model

    for name, mod in base.named_modules():
        if mod.__class__.__name__ not in ["SSM", "Progressive_SSM"]:
            continue

        with torch.no_grad():
            L = mod.Lambda.detach().float()           # [N, 2]  (re, im bruts)
            log_step = mod.log_step.detach().float()  # [N]
            step = (mod.step_scale * torch.exp(log_step))  # Delta  [N]

            if mod.ensure_stability == 'sigmoid_interval':
                width = mod.re_upper - mod.re_lower
                ld_re = mod.re_lower + width * torch.sigmoid(mod.sigmoid_scale * L[:, 0])
            else:
                ld_re = L[:, 0] * step

            ld_im = L[:, 1] * step  # fréquence en rad/sample
            LD = torch.complex(ld_re, ld_im)

            Lambda_bar = torch.exp(LD)
            mag = Lambda_bar.abs()        # [N]

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
                "ld_im_q1":  float(torch.quantile(ld_im, 0.25)) if not n_nan else None,
                "ld_im_q3":  float(torch.quantile(ld_im, 0.75)) if not n_nan else None,
                "ld_im_med":  float(torch.median(ld_im)) if not n_nan else None,
                "lbar_mag_mean": float(mag.mean()) if not n_nan else None,
                "lbar_mag_max":  float(mag.max())  if not n_nan else None,
                "loss_regularization": float(loss_regularization.item()) if not n_nan else None,
                "n_unstable":  int((ld_re > 0.0).sum())    if not n_nan else -1,
                "n_near_zero": int((ld_re > -1e-3).sum())  if not n_nan else -1,
                "n_aliasing" : int((ld_im > np.pi).sum())  if not n_nan else -1,
                "n_nan_params": int(torch.isnan(L).sum()),
            }
    return stats


def L2_im_lambda(model, alpha, beta):
    try:
        import model_edge
    except ImportError:
        return 0.0

    base = model.module if hasattr(model, 'module') else model
    penalties = []

    for name, mod in base.named_modules():
        if mod.__class__.__name__ not in ["SSM", "Progressive_SSM"]:
            continue

        L = mod.Lambda.float()                             # [N, 2]
        log_step = mod.log_step.float()                    # [N]
        step = mod.step_scale * torch.exp(log_step)        # Delta [N]

        Lambda_c = torch.complex(L[:, 0], L[:, 1])
        LD = Lambda_c * step                               # Lambda * Delta [N]
        ld_im = LD.imag                                    # rad/sample [N]

        excess_pos = torch.nn.functional.relu(ld_im - torch.pi)
        w_pos = excess_pos + beta * (excess_pos > 0).float()

        excess_neg = torch.nn.functional.relu(-ld_im)
        w_neg = excess_neg + beta * (excess_neg > 0).float()

        penalties.append((w_pos.pow(2) + w_neg.pow(2)).mean())

    if not penalties:
        return 0.0

    return alpha * torch.stack(penalties).mean()


def train(args, trainable_decoder, encoder, device, train_sampler, optimizer,
          is_distributed=False, scaler=None, ds=1, alpha=1e-2, beta=0):
    losses = utils.AverageMeter()
    nan_batches = 0
    trainable_decoder.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, _ in pbar:
        pbar.set_description("Training batch")
        x = x.to(device)
        if ds > 1:
            x = x[:, :, ::ds]

        optimizer.zero_grad()

        use_amp = (scaler is not None) and (device.type == "cuda")

        with amp_autocast(enabled=False):
            X = encoder(x.float())

        with amp_autocast(enabled=use_amp):
            y_hat = trainable_decoder(X, length = x.data.shape[-1])
            Y_hat = encoder(y_hat)
            # loss = torch.nn.functional.mse_loss(y_hat, x) + L2_im_lambda(trainable_decoder, alpha, beta)
            loss = torch.nn.functional.mse_loss(Y_hat, X) + L2_im_lambda(trainable_decoder, alpha, beta)


        if not torch.isfinite(loss):
            nan_batches += 1
            if not args.quiet:
                print_rank0(f"[WARN] Loss non-finie ({loss.item():.4g}) sur ce batch — batch ignore.")
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_decoder.parameters(), max_norm=1.0)
            optimizer.step()

        losses.update(loss.item(), x.size(0))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))

    if nan_batches > 0:
        print_rank0(f"[WARN] {nan_batches} batch(s) ignores (NaN/Inf) durant cette epoque.")

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


def valid(args, trainable_decoder, encoder, device, valid_sampler,
          is_distributed=False, use_amp=False, ds=1, alpha=1e-2, beta=0):
    losses = utils.AverageMeter()
    trainable_decoder.eval()
    X, Y_hat = None, None
    with torch.no_grad():
        for x, _ in valid_sampler:
            x = x.to(device)
            if ds > 1:
                x = x[:, :, ::ds]

            with amp_autocast(enabled=use_amp and (device.type == "cuda")):
                X = encoder(x)
                y_hat = trainable_decoder(X, length = x.data.shape[-1])
                Y_hat = encoder(y_hat)
                # loss = torch.nn.functional.mse_loss(y_hat, x) + L2_im_lambda(trainable_decoder, alpha, beta)
                loss = torch.nn.functional.mse_loss(Y_hat, X) + L2_im_lambda(trainable_decoder, alpha, beta)


            if not torch.isfinite(loss):
                continue
            losses.update(loss.item(), x.size(0))

    if is_distributed:
        loss_sum = torch.tensor(losses.sum if losses.count > 0 else 0.0, device=device)
        count_sum = torch.tensor(float(losses.count), device=device)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / count_sum).item() if count_sum.item() > 0 else float('nan')
    else:
        avg_loss = losses.avg if losses.count > 0 else float('nan')

    return avg_loss, X, Y_hat


def main():
    parser = argparse.ArgumentParser(description="Open trainable_decoder Distributed Trainer")

    # Target
    parser.add_argument("--target", type=str, default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset parameters
    parser.add_argument("--ds", type=int, default=1, help="downsampling factor for the input data")
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
        default="open-trainable_decoder",
        help="provide output path base folder name",
    )
    parser.add_argument("--model", type=str, help="Name or path of pretrained model to fine-tune")
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to resume training")
    parser.add_argument("--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument("--patience",
        type=int,
        default=140,
        help="maximum number of train epochs (default: 140)",
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
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")

    parser.add_argument("--alpha", type=float, default=0,
        help="factor that multiplies the L2 loss of the imaginary parts of the eigenvalues of the MAGSSM. No regularization per default")
    parser.add_argument("--beta", type=float, default=1,
        help="offset on the imaginary parts (frequencies) of the eigenvalues of the MAGSSM that are above pi if one wishes to have a stricter boundary.")

    parser.add_argument("--eps-stability", type=float, default=1e-3,
        help="Stability margin epsilon used in Progressive_SSM.forward() to clamp eigenvalue real parts "
             "away from 0. Smaller = tighter stability constraint. (default: 1e-3)")
    parser.add_argument("--dt-min", type=float, default=0.001,
        help="Minimum timescale (dt) for SSM log-step initialization. (default: 1e-3)")
    parser.add_argument("--dt-max", type=float, default=0.1,
        help="Maximum timescale (dt) for SSM log-step initialization. (default: 0.1)")

    parser.add_argument("--ensure-stability", type=str, default="abs",
        choices=["abs", "relu", "sigmoid_interval"],
        help="Stability enforcement mode for eigenvalue real parts. "
             "'abs'/'relu': legacy in-place clamping. "
             "'sigmoid_interval': differentiable sigmoid reparametrisation. (default: abs)")
    parser.add_argument("--re-lower", type=float, default=None,
        help="Lower bound for Re(Lambda)*step product when using --ensure-stability sigmoid_interval. "
             "Must be negative. Example: -0.00316 (= -10^(-2.5))")
    parser.add_argument("--re-upper", type=float, default=None,
        help="Upper bound for Re(Lambda)*step product when using --ensure-stability sigmoid_interval. "
        "Must be negative and > re_lower. Example: -3.16e-05 (= -10^(-4.5))")
    parser.add_argument("--sigmoid-scale", type=float, default=1.0,
        help="Internal scaling multiplier S inside the sigmoid for sigmoid_interval reparametrisation. "
             "S = 1.0 (default) behaves normally. Higher values accelerate eigenvalue updates.")

    # Model Parameters
    parser.add_argument("--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds — value of <=0.0 will use full/variable length",
    )
    parser.add_argument("--nfft", type=int, default=4096, help="(STFT) fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="(STFT) hop size")
    parser.add_argument("--nb_magssm_states", type=int, default=129, help="Number of states in MAGSSM")
    parser.add_argument("--d_out", type=int, default=None,
        help="Number of frequencies in the trainable spectrogram.")
    parser.add_argument("--chunk-dur", type=float, default=6.0,
        help="chunk duration in seconds. The input sequence will be split into chunks for computation by the SSM module")
    parser.add_argument("--progressive", action="store_true", default=False,
        help="If set, will proceed the SSM in chunks + gradient checkpointing, thus reducing VRAM consumption")
    parser.add_argument("--fft_kernel", "--fft-kernel", action="store_true", default=False,
        help="If set, will use Fast Fourier Transform convolution kernel for SSM computation")
    parser.add_argument("--mel", action="store_true", default=False,
        help="If set, will initialize eigenvalue arguments of the A-matrix on a log scale.")
    parser.add_argument("--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz")
    parser.add_argument("--nb-channels", type=int, default=2, help="set number of channels for model (1, 2)")
    parser.add_argument("--nb-workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--debug", action="store_true", default=False, help="Speed up training init for dev purposes")

    parser.add_argument("--hidden_size_factors", type=int, nargs="+", default=None)
    parser.add_argument("--output_size_factors", type=int, nargs="+", default=None)

    # Distributed Training Parameters
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use (default: nccl)")
    parser.add_argument("--master-addr", type=str, default=None,
                        help="Adresse IP ou hostname du noeud master (rank 0).")
    parser.add_argument("--master-port", type=str, default="29500",
                        help="Port TCP du noeud master (défaut: 29500).")

    # Misc Parameters
    parser.add_argument("--quiet", action="store_true", default=False, help="less verbose during training")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--amp", action="store_true", default=False, help="Use automatic mixed precision (AMP) during training")
    parser.add_argument("--og", action="store_true", default=False, help="Uses initialization of eigenvalues and B matrix from the original MagSSM paper")
    parser.add_argument("--structured_initialisation", action="store_true", default=False, help="Linearly spaced imaginary parts of the eigenvalues between 0 and pi.")
    parser.add_argument("--phase-correction", action="store_true", default=False,
        help="Apply phase correction to C for LFI decoder init.")

    parser.add_argument("--regularize_window", action="store_true", default=False)
    parser.add_argument("--epsilon1", type=float, default=0.07)
    parser.add_argument("--lambda_coeff_1", type=float, default=0.77)
    parser.add_argument("--lambda_coeff_2", type=float, default=0.85)

    args, _ = parser.parse_known_args()

    # DDP init
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        if args.master_addr is not None:
            os.environ["MASTER_ADDR"] = args.master_addr
        if "MASTER_ADDR" not in os.environ:
            raise RuntimeError(
                "MASTER_ADDR non défini. Utilisez --master-addr ou la variable d'environnement MASTER_ADDR."
            )
        os.environ.setdefault("MASTER_PORT", args.master_port)

        global_rank = int(os.environ["RANK"])
        local_rank  = int(os.environ["LOCAL_RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        global_rank = 0
        local_rank  = 0
        world_size  = 1

    args.quiet = args.quiet or (global_rank != 0)

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print_rank0("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    if not is_distributed:
        device = torch.device("cuda" if use_cuda else "cpu")

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        repo = Repo(repo_dir)
        commit = repo.head.commit.hexsha[:7]
    except Exception:
        commit = "unknown"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if is_distributed:
        if global_rank == 0:
            train_dataset, valid_dataset, args = data.load_datasets(parser, args)
        dist.barrier()
        if global_rank != 0:
            train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    else:
        train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    args.sample_rate = train_dataset.sample_rate // args.ds


    target_path = Path(args.output)
    if global_rank == 0:
        target_path.mkdir(parents=True, exist_ok=True)

    if is_distributed:
        train_sampler_ddp = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        valid_sampler_ddp = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
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

    stft, _ = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate,
        regularize=args.regularize_window, epsilon1=args.epsilon1,
        lambda_coeff_1=args.lambda_coeff_1, lambda_coeff_2=args.lambda_coeff_2,
    )

    window = transforms.get_regularised_window(n_fft=args.nfft, epsilon1=args.epsilon1,
        lambda_coeff_1=args.lambda_coeff_1, lambda_coeff_2=args.lambda_coeff_2,)

    encoder = stft.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate // args.ds,
        "nb_channels": args.nb_channels,
        "nb_magssm_states": args.nb_magssm_states,
        "regularize_window": args.regularize_window,
        "epsilon1": args.epsilon1,
        "lambda_coeff_1": args.lambda_coeff_1,
        "lambda_coeff_2": args.lambda_coeff_2,
    }

    if global_rank == 0:
        with open(Path(target_path, "separator.json"), "w") as outfile:
            outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.model:
        print_rank0(f"Fine-tuning model from {args.model}")
        trainable_decoder = utils_decoder.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True
        )[args.target]
        trainable_decoder = trainable_decoder.to(device)

        src_model_path = Path(args.model).expanduser()
        with open(Path(src_model_path, args.target + ".json"), "r") as f:
            src_results = json.load(f)
        src_best_epoch = src_results.get("best_epoch", 0)
        ft_init_train_losses = src_results.get("train_loss_history", [])[:src_best_epoch]
        ft_init_valid_losses = src_results.get("valid_loss_history", [])[:src_best_epoch]
        ft_init_train_times  = src_results.get("train_time_history",  [])[:src_best_epoch]
        ft_init_lambda_history = []
        src_lambda_path = Path(src_model_path, args.target + "_lambda.json")
        if src_lambda_path.exists():
            with open(src_lambda_path, "r") as f:
                src_lambda = json.load(f)
            ft_init_lambda_history = [e for e in src_lambda if e.get("epoch", 0) <= src_best_epoch]

    else:
        chunk_duration_in_frames = int(args.chunk_dur * args.sample_rate)
        progressive_mode = "fft" if args.fft_kernel else (True if args.progressive else False)

        trainable_decoder = Trainable_decoder(
            sample_rate=args.sample_rate,
            n_fft=args.nfft,
            n_hop=args.nhop,
            window = window,
            dim_state=args.nb_magssm_states,
            og=args.og,
            B_C_init="orthogonal",
            device=device,
            progressive=progressive_mode,
            chunk_duration=chunk_duration_in_frames,
            log_distributed_frequencies=args.mel,
            eps_stability=args.eps_stability,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            re_lower=args.re_lower,
            re_upper=args.re_upper,
            ensure_stability=args.ensure_stability,
            sigmoid_scale=args.sigmoid_scale,
            structured_initialisation=args.structured_initialisation,
        ).to(device)

        print_rank0("Proceeding by chunks: ", trainable_decoder.magssm_decoder.mimo.progressive)
        total_params = sum(p.numel() for p in trainable_decoder.parameters() if p.requires_grad)
        print_rank0(f"Total number of parameters: {total_params}")

    trainable_decoder.alpha = args.alpha
    trainable_decoder.beta = args.beta

    optimizer = torch.optim.AdamW(
        trainable_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    amp_scaler = amp_grad_scaler(enabled=args.amp and use_cuda)

    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        trainable_decoder.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times  = results["train_time_history"]
        best_epoch   = results["best_epoch"]
        es.best          = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        if args.model:
            train_losses = ft_init_train_losses
            valid_losses = ft_init_valid_losses
            train_times  = ft_init_train_times
        else:
            train_losses = []
            valid_losses = []
            train_times  = []
        best_epoch = 0

    if is_distributed:
        trainable_decoder = torch.nn.parallel.DistributedDataParallel(
            trainable_decoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    lambda_log_path = Path(target_path, args.target + "_lambda.json")
    if lambda_log_path.exists():
        with open(lambda_log_path) as f:
            lambda_history = json.load(f)
    elif args.model and not args.checkpoint:
        lambda_history = ft_init_lambda_history
    else:
        lambda_history = []

    stft_best_path   = Path(target_path, args.target + "_stft_best.png")   if global_rank == 0 else None
    trainable_decoder_best_path = Path(target_path, args.target + "_magssm_best.png") if global_rank == 0 else None

    for epoch in t:
        if is_distributed:
            train_sampler_ddp.set_epoch(epoch)

        t.set_description("Training epoch")
        end = time.time()

        train_loss = train(
            args, trainable_decoder, encoder, device, train_sampler, optimizer,
            is_distributed=is_distributed, scaler=amp_scaler, ds=args.ds,
            alpha=args.alpha, beta=args.beta
        )
        valid_loss, X_val, Y_hat_val = valid(
            args, trainable_decoder, encoder, device, valid_sampler,
            is_distributed=is_distributed, use_amp=args.amp, ds=args.ds,
            alpha=args.alpha, beta=args.beta
        )

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        is_best = valid_loss == es.best
        if is_best:
            best_epoch = epoch

        if global_rank == 0:
            if is_best:
                if stft_best_path is not None and X_val is not None:
                    save_spectrogram(X_val, stft_best_path, complex_spectrogram=True)
                if trainable_decoder_best_path is not None and Y_hat_val is not None:
                    save_spectrogram(Y_hat_val, trainable_decoder_best_path, complex_spectrogram=True)

            model_to_save = trainable_decoder.module if hasattr(trainable_decoder, 'module') else trainable_decoder
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                    "best_loss": es.best,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best=is_best,
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

            lambda_stats = collect_lambda_stats(trainable_decoder)
            lambda_history.append({"epoch": epoch, "lambda": lambda_stats})
            with open(lambda_log_path, "w") as f:
                json.dump(lambda_history, f, indent=2)

            for mod_name, s in lambda_stats.items():
                if s.get("n_nan_params", 0) > 0:
                    print_rank0(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: NaN dans Lambda ({s['n_nan_params']} params)")
                elif s.get("n_unstable", 0) > 0:
                    print_rank0(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: {s['n_unstable']} états instables (Re(Λ·Δ)>0), ld_re_max={s['ld_re_max']:.3e}")
                elif s.get("ld_re_max") is not None and s["ld_re_max"] > -1e-3:
                    print_rank0(f"[Lambda] ! EPOCH {epoch} — {mod_name}: ld_re_max={s['ld_re_max']:.3e} (Re(Λ·Δ) proche de 0)")

        train_times.append(time.time() - end)

        if stop:
            print_rank0("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
