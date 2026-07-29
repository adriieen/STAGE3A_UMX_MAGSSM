import argparse
import os
import torch
import torch.distributed as dist

# Set CUDA device ASAP, before importing anything else (like torchaudio)
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
import pylab

import data
import model
import utils
import transforms
import magssm_umx
from decoder import Trainable_decoder
from spectrogram import Trainable_spectrogram
import utils_spectrogram
import utils_decoder
from path_config import setup_paths, amp_autocast, amp_grad_scaler
setup_paths()

tqdm.monitor_interval = 0


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def save_spectrogram(X, save_path: str, complex_spectrogram=True):
    """
    Saves the spectrogram of a wavefile of format 1=batch, C=2, F, T in a .png file.
    Handles 5D complex tensors (B, C, F, T, 2) automatically.
    """
    if X.ndim == 5:
        X_re, X_im = X[..., 0], X[..., 1]
        X = X_re + 1j * X_im
    elif torch.is_complex(X):
        pass
    else:
        pass

    X = torch.mean(X, dim=1)
    X = torch.abs(X[0])
    X = torch.log(X + 1e-8)
    pylab.imshow(X.detach().cpu().numpy(), aspect='auto', origin='lower')
    pylab.tight_layout()
    pylab.savefig(save_path, dpi=300)
    pylab.close()


# ---------------------------------------------------------------------------
# Tracking of SSM Lambda eigenvalues
# ---------------------------------------------------------------------------

def collect_lambda_stats(model_or_module) -> dict:
    """
    Traverses all Progressive_SSM layers inside the model and collects
    statistics on Lambda * Delta.
    """
    try:
        import model_edge
    except ImportError:
        return {}

    stats = {}
    base = model_or_module.module if hasattr(model_or_module, 'module') else model_or_module

    for name, mod in base.named_modules():
        if mod.__class__.__name__ not in ["SSM", "Progressive_SSM", "Progressive_SSM_FT"]:
            continue

        with torch.no_grad():
            L = mod.Lambda.detach().float()           # [N, 2]
            log_step = mod.log_step.detach().float()  # [N]
            step = (mod.step_scale * torch.exp(log_step))  # Delta [N]

            if getattr(mod, 'ensure_stability', None) == 'sigmoid_interval':
                width = mod.re_upper - mod.re_lower
                ld_re = mod.re_lower + width * torch.sigmoid(mod.sigmoid_scale * L[:, 0])
            else:
                ld_re = L[:, 0] * step

            ld_im = L[:, 1] * step
            LD = torch.complex(ld_re, ld_im)

            Lambda_bar = torch.exp(LD)
            mag = Lambda_bar.abs()

            n_nan = torch.isnan(L).any().item()

            alpha = getattr(base, 'alpha', 0.0)
            beta = getattr(base, 'beta', 0.0)

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
                "loss_regularization": float(loss_regularization.item()) if not n_nan else None,
                "n_unstable":  int((ld_re > 0.0).sum())    if not n_nan else -1,
                "n_near_zero": int((ld_re > -1e-3).sum())  if not n_nan else -1,
                "n_nan_params": int(torch.isnan(L).sum()),
            }
    return stats


def L2_im_lambda(model_or_module, alpha, beta):
    if alpha <= 0:
        return 0.0

    base = model_or_module.module if hasattr(model_or_module, 'module') else model_or_module
    penalties = []

    for name, mod in base.named_modules():
        if mod.__class__.__name__ not in ["SSM", "Progressive_SSM", "Progressive_SSM_FT"]:
            continue

        L = mod.Lambda.float()
        log_step = mod.log_step.float()
        step = mod.step_scale * torch.exp(log_step)

        Lambda_c = torch.complex(L[:, 0], L[:, 1])
        LD = Lambda_c * step
        ld_im = LD.imag

        excess_pos = torch.nn.functional.relu(ld_im - torch.pi)
        w_pos = excess_pos + beta * (excess_pos > 0).float()

        excess_neg = torch.nn.functional.relu(-ld_im)
        w_neg = excess_neg + beta * (excess_neg > 0).float()

        penalties.append((w_pos.pow(2) + w_neg.pow(2)).mean())

    if not penalties:
        return 0.0

    return alpha * torch.stack(penalties).mean()


def train(args, model, trainable_decoder, encoder, device, train_sampler, optimizer,
          is_distributed=False, scaler=None, ds=1, alpha=1e-2, beta=0):
    losses = utils.AverageMeter()
    nan_batches = 0
    model.train()
    trainable_decoder.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        if ds > 1:
            x = x[:, :, ::ds]
            y = y[:, :, ::ds]

        optimizer.zero_grad()
        use_amp = (scaler is not None) and (device.type == "cuda")

        # Compute ground-truth target STFT
        with amp_autocast(enabled=False):
            Y = encoder(y.float())

        with amp_autocast(enabled=use_amp):
            # 1. C-MAGSSM Spectrogram + OpenUnmix Encoder + Separator + OpenUnmix Decoder
            S_target_est = model(x)

            # 2. DEC-SSM Trainable decoder -> estimated audio waveform y_hat
            y_hat = trainable_decoder(S_target_est, length=x.data.shape[-1])

            # 3. Compute STFT of estimated audio
            Y_hat = encoder(y_hat.float())

            # 4. Loss in STFT domain + eigenvalue regularization
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            if alpha > 0:
                loss = loss + L2_im_lambda(model, alpha, beta) + L2_im_lambda(trainable_decoder, alpha, beta)

        if not torch.isfinite(loss):
            nan_batches += 1
            if not args.quiet:
                print_rank0(f"[WARN] Non-finite loss ({loss.item():.4g}) on this batch — batch ignored.")
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(trainable_decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(trainable_decoder.parameters(), max_norm=1.0)
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


def valid(args, model, trainable_decoder, encoder, device, valid_sampler,
          is_distributed=False, use_amp=False, ds=1, alpha=1e-2, beta=0):
    losses = utils.AverageMeter()
    model.eval()
    trainable_decoder.eval()
    Y_last, Y_hat_last = None, None

    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            if ds > 1:
                x = x[:, :, ::ds]
                y = y[:, :, ::ds]

            with amp_autocast(enabled=use_amp and (device.type == "cuda")):
                Y = encoder(y.float())
                S_target_est = model(x)
                y_hat = trainable_decoder(S_target_est, length=x.data.shape[-1])
                Y_hat = encoder(y_hat.float())

                loss = torch.nn.functional.mse_loss(Y_hat, Y)
                if alpha > 0:
                    loss = loss + L2_im_lambda(model, alpha, beta) + L2_im_lambda(trainable_decoder, alpha, beta)

            if not torch.isfinite(loss):
                continue
            losses.update(loss.item(), Y.size(1))
            Y_last, Y_hat_last = Y, Y_hat

    if is_distributed:
        loss_sum = torch.tensor(losses.sum if losses.count > 0 else 0.0, device=device)
        count_sum = torch.tensor(float(losses.count), device=device)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / count_sum).item() if count_sum.item() > 0 else float('nan')
    else:
        avg_loss = losses.avg if losses.count > 0 else float('nan')

    return avg_loss, Y_last, Y_hat_last


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
        "n_fft": conf.get("nfft", conf.get("n_fft", 4096)),
        "n_hop": conf.get("nhop", conf.get("n_hop", 1024)),
        "nb_magssm_states": conf.get("nb_magssm_states", 129),
        "d_out": conf.get("d_out", None),
        "og": conf.get("og", False),
        "structured_initialisation": conf.get("structured_initialisation", False),
        "eps_stability": conf.get("eps_stability", 1e-3),
        "dt_min": conf.get("dt_min", 0.001),
        "dt_max": conf.get("dt_max", 0.1),
        "regularize_window": True if str(conf.get("regularize_window")).lower() in ["true", "1"] else False,
        "epsilon1": conf.get("epsilon1", 0.07),
        "lambda_coeff_1": conf.get("lambda_coeff_1", 0.77),
        "lambda_coeff_2": conf.get("lambda_coeff_2", 0.85),
        "sample_rate": conf.get("sample_rate", 44100),
        "complex_spectrogram": conf.get("complex_spectrogram", True),
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
    parser = argparse.ArgumentParser(description="MagSSM OpenUnmix Combined Pipeline Fine-tuner")

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
        default="open-model-magssm",
        help="provide output path base folder name",
    )
    
    # Pre-trained models paths
    parser.add_argument("--model", type=str, default=None,
        help="Path of pre-trained OpenUnmix backbone checkpoint directory"
    )
    parser.add_argument("--ssm-model", type=str, default=None,
        help="Path of pre-trained trainable spectrogram SSM checkpoint directory"
    )
    parser.add_argument("--decoder-model", type=str, default=None,
        help="Path of pre-trained trainable decoder SSM checkpoint directory"
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
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
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

    parser.add_argument("--alpha", type=float, default=0,
        help="factor that multiplies the L2 loss of the imaginary parts of the eigenvalues of the MAGSSM.")
    parser.add_argument("--beta", type=float, default=1,
        help="offset on the imaginary parts of the eigenvalues above pi.")

    parser.add_argument("--eps-stability", type=float, default=1e-3,
        help="Stability margin epsilon used in Progressive_SSM.forward()")
    parser.add_argument("--dt-min", type=float, default=0.001,
        help="Minimum timescale (dt) for SSM log-step initialization")
    parser.add_argument("--dt-max", type=float, default=0.1,
        help="Maximum timescale (dt) for SSM log-step initialization")

    parser.add_argument("--ensure-stability", type=str, default="abs",
        choices=["abs", "relu", "sigmoid_interval"],
        help="Stability enforcement mode for eigenvalue real parts.")
    parser.add_argument("--re-lower", type=float, default=None,
        help="Lower bound for Re(Lambda)*step product when using sigmoid_interval.")
    parser.add_argument("--re-upper", type=float, default=None,
        help="Upper bound for Re(Lambda)*step product when using sigmoid_interval.")
    parser.add_argument("--sigmoid-scale", type=float, default=1.0,
        help="Internal scaling multiplier S inside the sigmoid.")

    # Freezing and architecture customization
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
        help="Freeze the OpenUnmix separation backbone and only optimize the SSM encoder & decoder parameters"
    )
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone",
        help="Do not freeze the separation backbone"
    )
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
        help="Learning rate for backbone parameters when not frozen"
    )
    parser.add_argument("--freeze-ssm", action="store_true", default=False,
        help="Freeze C-MAGSSM encoder parameters"
    )
    parser.add_argument("--freeze-decoder", action="store_true", default=False,
        help="Freeze DEC-SSM decoder parameters"
    )

    # Model parameters override
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
    parser.add_argument("--nfft", type=int, default=4096, help="(STFT) fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="(STFT) hop size")
    parser.add_argument("--nb_magssm_states", type=int, default=129, help="Number of states in MAGSSM")
    parser.add_argument("--d_out", type=int, default=None, help="Number of frequencies in trainable spectrogram")
    parser.add_argument("--progressive", action="store_true", default=False, help="Process SSM in chunks + gradient checkpointing")
    parser.add_argument("--fft_kernel", action="store_true", default=False, help="Process SSM with FFT kernel")
    parser.add_argument("--chunk-dur", type=float, default=6.0, help="chunk duration in seconds")
    parser.add_argument("--mel", action="store_true", default=False, help="Initialize eigenvalues on log scale")
    parser.add_argument("--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz")
    parser.add_argument("--nb-channels", type=int, default=2, help="set number of channels for model")
    parser.add_argument("--nb-workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--debug", action="store_true", default=False, help="Speed up training init for dev purposes")
    parser.add_argument("--quiet", action="store_true", default=False, help="less verbose during training")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--amp", action="store_true", default=False, help="Use automatic mixed precision (AMP)")
    parser.add_argument("--complex_spectrogram", action="store_true", default=True, help="Use complex spectrograms")
    parser.add_argument("--structured_initialisation", action="store_true", default=False, help="Linearly spaced imaginary parts")
    parser.add_argument("--og", action="store_true", default=False, help="Original MagSSM initialization")
    parser.add_argument("--phase-correction", action="store_true", default=False, help="Apply phase correction to C for LFI decoder")

    parser.add_argument("--regularize_window", action="store_true", default=False)
    parser.add_argument("--epsilon1", type=float, default=0.07)
    parser.add_argument("--lambda_coeff_1", type=float, default=0.77)
    parser.add_argument("--lambda_coeff_2", type=float, default=0.85)

    # Distributed Training Parameters
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use (default: nccl)")
    parser.add_argument("--master-addr", type=str, default=None,
                        help="Adresse IP ou hostname du noeud master (rank 0).")
    parser.add_argument("--master-port", type=str, default="29500",
                        help="Port TCP du noeud master (défaut: 29500).")

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

    # 1. Detect SSM parameters from pretrained SSMs if provided
    n_fft = args.nfft
    n_hop = args.nhop
    nb_magssm_states = args.nb_magssm_states
    d_out = args.d_out

    ssm_conf = None
    dec_conf = None
    encoder_nb_states = args.nb_magssm_states
    decoder_nb_states = args.nb_magssm_states

    if args.ssm_model:
        ssm_conf = load_ssm_params(args.ssm_model)
        n_fft = ssm_conf["n_fft"]
        n_hop = ssm_conf["n_hop"]
        encoder_nb_states = ssm_conf["nb_magssm_states"]
        args.og = ssm_conf.get("og", args.og)
        args.structured_initialisation = ssm_conf.get("structured_initialisation", args.structured_initialisation)
        args.eps_stability = ssm_conf.get("eps_stability", args.eps_stability)
        args.dt_min = ssm_conf.get("dt_min", args.dt_min)
        args.dt_max = ssm_conf.get("dt_max", args.dt_max)
        args.mel = ssm_conf.get("mel", args.mel)
        args.complex_spectrogram = ssm_conf.get("complex_spectrogram", True)

        assert args.complex_spectrogram is True, \
            "ERREUR : Le modèle C-MAGSSM pré-entraîné fourni n'est pas un spectrogramme complexe (complex_spectrogram=False). La pipeline combinée exige complex_spectrogram=True."

        print_rank0(f"[SSM Encoder Auto-Detected] n_fft={n_fft}, n_hop={n_hop}, encoder_states={encoder_nb_states}, og={args.og}, complex_spectrogram={args.complex_spectrogram}")

    if args.decoder_model:
        dec_conf = load_ssm_params(args.decoder_model)
        decoder_nb_states = dec_conf["nb_magssm_states"]
        args.phase_correction = dec_conf.get("phase_correction", args.phase_correction)
        print_rank0(f"[SSM Decoder Auto-Detected] n_fft={dec_conf['n_fft']}, n_hop={dec_conf['n_hop']}, decoder_states={decoder_nb_states}, phase_correction={args.phase_correction}")

    # Check fundamental STFT and window consistency between pretrained SSM encoder and decoder
    if ssm_conf is not None and dec_conf is not None:
        assert ssm_conf["n_fft"] == dec_conf["n_fft"], \
            f"Incompatibilité n_fft entre l'encodeur ({ssm_conf['n_fft']}) et le décodeur ({dec_conf['n_fft']})"
        assert ssm_conf["n_hop"] == dec_conf["n_hop"], \
            f"Incompatibilité n_hop entre l'encodeur ({ssm_conf['n_hop']}) et le décodeur ({dec_conf['n_hop']})"
        assert ssm_conf["regularize_window"] == dec_conf["regularize_window"], \
            f"Incompatibilité regularize_window : encodeur={ssm_conf['regularize_window']} vs décodeur={dec_conf['regularize_window']}"

        if ssm_conf["regularize_window"]:
            assert np.isclose(ssm_conf["epsilon1"], dec_conf["epsilon1"], atol=1e-4), \
                f"Incompatibilité epsilon1 : encodeur={ssm_conf['epsilon1']} vs décodeur={dec_conf['epsilon1']}"
            assert np.isclose(ssm_conf["lambda_coeff_1"], dec_conf["lambda_coeff_1"], atol=1e-4), \
                f"Incompatibilité lambda_coeff_1 : encodeur={ssm_conf['lambda_coeff_1']} vs décodeur={dec_conf['lambda_coeff_1']}"
            assert np.isclose(ssm_conf["lambda_coeff_2"], dec_conf["lambda_coeff_2"], atol=1e-4), \
                f"Incompatibilité lambda_coeff_2 : encodeur={ssm_conf['lambda_coeff_2']} vs décodeur={dec_conf['lambda_coeff_2']}"
            print_rank0(f"[Window Consistency Check] ✓ Fenêtre régularisée identique vérifiée entre l'encodeur et le décodeur (eps1={ssm_conf['epsilon1']}, l1={ssm_conf['lambda_coeff_1']}, l2={ssm_conf['lambda_coeff_2']})")

    # Inherit window regularization parameters from pretrained models if not overridden
    if ssm_conf is not None and ssm_conf.get("regularize_window"):
        args.regularize_window = True
        args.epsilon1 = ssm_conf["epsilon1"]
        args.lambda_coeff_1 = ssm_conf["lambda_coeff_1"]
        args.lambda_coeff_2 = ssm_conf["lambda_coeff_2"]

    d_out = n_fft // 2 + 1

    # STFT filterbanks & regularized window setup
    stft, _ = transforms.make_filterbanks(
        n_fft=n_fft, n_hop=n_hop, sample_rate=train_dataset.sample_rate,
        regularize=args.regularize_window, epsilon1=args.epsilon1,
        lambda_coeff_1=args.lambda_coeff_1, lambda_coeff_2=args.lambda_coeff_2,
    )

    if args.regularize_window:
        window = transforms.get_regularised_window(
            n_fft=n_fft, epsilon1=args.epsilon1,
            lambda_coeff_1=args.lambda_coeff_1, lambda_coeff_2=args.lambda_coeff_2,
        )
    else:
        window = None

    # Reference STFT encoder for training loss computation
    if args.complex_spectrogram:
        encoder = stft.to(device)
    else:
        encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 2. Detect backbone parameters from pretrained OpenUnmix backbone if provided
    scaler_mean = None
    scaler_std = None
    has_ln = True
    hidden_size = 512
    nb_layers = 3

    if args.model:
        backbone_dir = Path(args.model).expanduser()
        backbone_file = next(backbone_dir.glob("%s*.chkpnt" % args.target), None)
        if not backbone_file:
            backbone_file = next(backbone_dir.glob("%s*.pth" % args.target), None)
        if backbone_file:
            print_rank0(f"Inspecting backbone weights from {backbone_file}...")
            backbone_state = torch.load(backbone_file, map_location="cpu", weights_only=False)
            state_dict_to_inspect = backbone_state.get("state_dict", backbone_state)
            has_ln = any("ln1" in k for k in state_dict_to_inspect.keys())
            hidden_size, nb_layers = load_backbone_params(args.model)

            if "input_mean" in state_dict_to_inspect:
                scaler_mean = -state_dict_to_inspect["input_mean"].numpy()
            if "input_scale" in state_dict_to_inspect:
                scaler_std = 1.0 / state_dict_to_inspect["input_scale"].numpy()

    chunk_duration_in_frames = int(args.chunk_dur * train_dataset.sample_rate)

    # 3. Instantiate combined model (C-MAGSSM + OpenUnmix Backbone)
    combined_model = magssm_umx.MagSSM_OpenUnmix(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_bins=n_fft // 2 + 1,
        nb_channels=args.nb_channels,
        hidden_size=hidden_size,
        nb_layers=nb_layers,
        unidirectional=args.unidirectional,
        dim_state=encoder_nb_states,
        d_out=d_out,
        n_fft=n_fft,
        n_hop=n_hop,
        encoder=encoder,
        device=device,
        chunk_duration=chunk_duration_in_frames,
        log_distributed_frequencies=args.mel,
        use_layernorm=has_ln,
        og=args.og,
        eps_stability=args.eps_stability,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        re_lower=args.re_lower,
        re_upper=args.re_upper,
        ensure_stability=args.ensure_stability,
        sigmoid_scale=args.sigmoid_scale,
        progressive=args.progressive,
        fft_kernel=args.fft_kernel,
        complex_spectrogram=args.complex_spectrogram,
        structured_initialisation=args.structured_initialisation,
    ).to(device)

    # 4. Instantiate trainable decoder (DEC-SSM)
    trainable_decoder = Trainable_decoder(
        sample_rate=train_dataset.sample_rate,
        n_fft=n_fft,
        n_hop=n_hop,
        window=window,
        dim_state=decoder_nb_states,
        og=args.og,
        B_C_init="orthogonal",
        device=device,
        progressive=args.progressive,
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
        phase_correction=args.phase_correction,
    ).to(device)

    combined_model.alpha = args.alpha
    combined_model.beta = args.beta
    trainable_decoder.alpha = args.alpha
    trainable_decoder.beta = args.beta

    # 5. Load pre-trained weights
    # A) Load SSM Encoder weights into combined_model.trainable_spectrogram
    if args.ssm_model:
        ssm_file = next(Path(args.ssm_model).expanduser().glob("%s*.chkpnt" % args.target), None)
        if not ssm_file:
            ssm_file = next(Path(args.ssm_model).expanduser().glob("%s*.pth" % args.target), None)
        if ssm_file:
            print_rank0(f"Loading pre-trained SSM encoder weights from {ssm_file}")
            ssm_state = torch.load(ssm_file, map_location=device, weights_only=False)
            ssm_state_dict = ssm_state.get("state_dict", ssm_state)
            ssm_filtered = {k: v for k, v in ssm_state_dict.items() if "magssm_encoder" in k or "trainable_spectrogram" in k}
            combined_model.load_state_dict(ssm_filtered, strict=False)

    # B) Load OpenUnmix Backbone weights into combined_model
    if args.model and backbone_file:
        print_rank0(f"Loading pre-trained backbone weights from {backbone_file}")
        backbone_filtered = {k: v for k, v in state_dict_to_inspect.items() if "magssm_encoder" not in k and "trainable_spectrogram" not in k and "encoder" not in k}
        combined_model.load_state_dict(backbone_filtered, strict=False)

    # C) Load Trainable Decoder weights into trainable_decoder
    if args.decoder_model:
        dec_file = next(Path(args.decoder_model).expanduser().glob("%s*.chkpnt" % args.target), None)
        if not dec_file:
            dec_file = next(Path(args.decoder_model).expanduser().glob("%s*.pth" % args.target), None)
        if dec_file:
            print_rank0(f"Loading pre-trained DEC-SSM decoder weights from {dec_file}")
            dec_state = torch.load(dec_file, map_location=device, weights_only=False)
            dec_state_dict = dec_state.get("state_dict", dec_state)
            trainable_decoder.load_state_dict(dec_state_dict, strict=False)

    # 6. Apply parameter freezing
    if args.freeze_backbone:
        for name, param in combined_model.named_parameters():
            if "trainable_spectrogram" in name or "magssm_encoder" in name:
                param.requires_grad = not args.freeze_ssm
            else:
                param.requires_grad = False
        print_rank0("[INFO] Backbone weights FROZEN.")
    else:
        for name, param in combined_model.named_parameters():
            if "trainable_spectrogram" in name or "magssm_encoder" in name:
                param.requires_grad = not args.freeze_ssm
            else:
                param.requires_grad = True
        print_rank0("[INFO] Backbone weights TRAINABLE.")

    if args.freeze_ssm:
        combined_model.trainable_spectrogram.freeze()
        print_rank0("[INFO] C-MAGSSM Spectrogram SSM FROZEN.")

    if args.freeze_decoder:
        trainable_decoder.freeze()
        print_rank0("[INFO] DEC-SSM Decoder FROZEN.")

    # Setup Optimizer
    ssm_params = [p for n, p in combined_model.named_parameters() if ("trainable_spectrogram" in n or "magssm_encoder" in n) and p.requires_grad]
    decoder_params = [p for p in trainable_decoder.parameters() if p.requires_grad]
    backbone_params = [p for n, p in combined_model.named_parameters() if ("trainable_spectrogram" not in n and "magssm_encoder" not in n) and p.requires_grad]

    all_trainable = ssm_params + decoder_params + backbone_params
    total_trainable = sum(p.numel() for p in all_trainable)
    print_rank0(f"Total number of trainable parameters: {total_trainable}")

    if args.freeze_backbone or not backbone_params:
        optimizer_groups = [
            {"params": ssm_params + decoder_params, "lr": args.lr}
        ]
    else:
        optimizer_groups = [
            {"params": ssm_params + decoder_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr_backbone}
        ]

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)
    amp_scaler = amp_grad_scaler(enabled=args.amp and use_cuda)

    # Resume from checkpoint
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        combined_model.load_state_dict(checkpoint["state_dict_model"], strict=False)
        trainable_decoder.load_state_dict(checkpoint["state_dict_decoder"], strict=False)
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
        train_losses = []
        valid_losses = []
        train_times  = []
        best_epoch   = 0

    # Wrap DDP
    if is_distributed:
        combined_model = torch.nn.parallel.DistributedDataParallel(
            combined_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
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
    else:
        lambda_history = []

    separator_conf = {
        "n_fft": n_fft,
        "n_hop": n_hop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
        "nb_magssm_states": nb_magssm_states,
        "use_layernorm": has_ln,
        "backbone_frozen": args.freeze_backbone,
        "regularize_window": args.regularize_window,
        "epsilon1": args.epsilon1,
        "lambda_coeff_1": args.lambda_coeff_1,
        "lambda_coeff_2": args.lambda_coeff_2,
        "complex_spectrogram": args.complex_spectrogram,
    }
    if global_rank == 0:
        with open(Path(target_path, "separator.json"), "w") as outfile:
            outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    stft_best_path = Path(target_path, args.target + "_stft_best.png") if global_rank == 0 else None
    magssm_best_path = Path(target_path, args.target + "_magssm_best.png") if global_rank == 0 else None

    # Training Loop
    for epoch in t:
        if is_distributed:
            train_sampler_ddp.set_epoch(epoch)

        t.set_description("Training epoch")
        end = time.time()

        train_loss = train(
            args, combined_model, trainable_decoder, encoder, device, train_sampler, optimizer,
            is_distributed=is_distributed, scaler=amp_scaler, ds=args.ds,
            alpha=args.alpha, beta=args.beta
        )
        valid_loss, Y_val, Y_hat_val = valid(
            args, combined_model, trainable_decoder, encoder, device, valid_sampler,
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
                if stft_best_path is not None and Y_val is not None:
                    save_spectrogram(Y_val, stft_best_path, complex_spectrogram=args.complex_spectrogram)
                if magssm_best_path is not None and Y_hat_val is not None:
                    save_spectrogram(Y_hat_val, magssm_best_path, complex_spectrogram=args.complex_spectrogram)

            model_to_save = combined_model.module if hasattr(combined_model, 'module') else combined_model
            decoder_to_save = trainable_decoder.module if hasattr(trainable_decoder, 'module') else trainable_decoder

            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                    "state_dict_model": model_to_save.state_dict(),
                    "state_dict_decoder": decoder_to_save.state_dict(),
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

            lambda_stats_model = collect_lambda_stats(combined_model)
            lambda_stats_dec = collect_lambda_stats(trainable_decoder)
            lambda_history.append({"epoch": epoch, "model_lambda": lambda_stats_model, "decoder_lambda": lambda_stats_dec})
            with open(lambda_log_path, "w") as f:
                json.dump(lambda_history, f, indent=2)

        train_times.append(time.time() - end)

        if stop:
            print_rank0("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
