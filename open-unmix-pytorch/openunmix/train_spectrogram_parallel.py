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
import shutil


import data
import model
import utils
import transforms
from spectrogram import Trainable_spectrogram
import utils_edge_var
from path_config import amp_autocast, amp_grad_scaler


tqdm.monitor_interval = 0

import pylab

def save_spectrogram(X, save_path : str):
    """
    Saves the spectrogram of a wavefile of format 1=batch, C=2, F, T in a .png file.
    """

    X = torch.mean(X,dim=1)
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
    pour chaque module, les statistiques du produit Lambda * Delta
    (exposant de la discrétisation ZOH : Lambda_bar = exp(Lambda * Delta)).

    Re(Lambda * Delta) = taux d'amortissement effectif par pas de temps
                         → doit rester < 0 pour la stabilité
    Im(Lambda * Delta) = fréquence d'oscillation en rad/sample
                         → détermine la sélectivité fréquentielle de l'état

    Retourne un dict JSON-serialisable :
      {
        "<module_name>": {
            "ld_re_mean", "ld_re_min", "ld_re_max",   # Re(Λ·Δ)
            "ld_im_mean", "ld_im_min", "ld_im_max",   # Im(Λ·Δ)
            "lbar_mag_mean", "lbar_mag_max",           # |Lambda_bar| = exp(Re(Λ·Δ))
            "n_unstable",    # nb états avec Re(Λ·Δ) > 0  ↔  |Λbar| > 1
            "n_near_zero",   # nb états avec Re(Λ·Δ) > -1e-3  (quasi-instables)
            "n_nan_params",
        }, ...
      }
    """
    try:
        from model_edge.ssm_bis import Progressive_SSM
    except ImportError:
        return {}

    stats = {}
    # Si le modèle est wrappé (DDP), accéder au module sous-jacent
    base = model.module if hasattr(model, 'module') else model

    for name, mod in base.named_modules():
        if not isinstance(mod, Progressive_SSM):
            continue

        with torch.no_grad():
            L = mod.Lambda.detach().float()           # [N, 2]  (re, im bruts)
            log_step = mod.log_step.detach().float()  # [N]
            step = (mod.step_scale * torch.exp(log_step))  # Delta  [N]

            # Produit Lambda * Delta — c'est l'exposant physiquement signifiant
            Lambda_c = torch.complex(L[:, 0], L[:, 1])
            LD = Lambda_c * step          # Lambda * Delta  [N] complexe
            ld_re = LD.real               # taux d'amortissement effectif
            ld_im = LD.imag               # fréquence en rad/sample

            # Lambda_bar = exp(Lambda * Delta)  →  |Lambda_bar| = exp(Re(Λ·Δ))
            Lambda_bar = torch.exp(LD)
            mag = Lambda_bar.abs()        # [N]

            n_nan = torch.isnan(L).any().item()

            stats[name] = {
                # Re(Lambda * Delta) — amortissement effectif
                "ld_re_mean": float(ld_re.mean()) if not n_nan else None,
                "ld_re_min":  float(ld_re.min())  if not n_nan else None,
                "ld_re_max":  float(ld_re.max())  if not n_nan else None,
                # Im(Lambda * Delta) — fréquence en rad/sample
                "ld_im_mean": float(ld_im.mean()) if not n_nan else None,
                "ld_im_min":  float(ld_im.min())  if not n_nan else None,
                "ld_im_max":  float(ld_im.max())  if not n_nan else None,
                # |Lambda_bar| = exp(Re(Λ·Δ))
                "lbar_mag_mean": float(mag.mean()) if not n_nan else None,
                "lbar_mag_max":  float(mag.max())  if not n_nan else None,
                # Indicateurs de danger
                "n_unstable":  int((ld_re > 0.0).sum())    if not n_nan else -1,
                "n_near_zero": int((ld_re > -1e-3).sum())  if not n_nan else -1,
                "n_nan_params": int(torch.isnan(L).sum()),
            }
    return stats


def train(args, trainable_spectrogram, encoder, device, train_sampler, optimizer,
          is_distributed=False, scaler=None, ds =1):
    losses = utils.AverageMeter()
    nan_batches = 0
    trainable_spectrogram.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, _ in pbar:
        pbar.set_description("Training batch") # B,2,L
        x = x.to(device)
        optimizer.zero_grad()
        x = x[:,:,::ds]

        use_amp = (scaler is not None) and (device.type == "cuda")

        # Precompute STFT completely outside of autocast and DDP
        with amp_autocast(enabled=False):
            X = encoder(x.float())

        with amp_autocast(enabled=use_amp):
            X_hat = trainable_spectrogram(x)
            loss = torch.nn.functional.mse_loss(X_hat, X)

        if not torch.isfinite(loss):
            nan_batches += 1
            if not args.quiet:
                print_rank0(f"[WARN] Loss non-finie ({loss.item():.4g}) sur ce batch — batch ignore.")
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_spectrogram.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_spectrogram.parameters(), max_norm=1.0)
            optimizer.step()

        losses.update(loss.item(), X.size(1))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))

    if nan_batches > 0:
        print_rank0(f"[WARN] {nan_batches} batch(s) ignores (NaN/Inf) durant cette epoque.")

    # Agréger correctement en ignorant les rangs sans batches valides
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


def valid(args, trainable_spectrogram, encoder, device, valid_sampler,
          is_distributed=False, use_amp=False, ds = 1, stft_saving_path=None, magssm_saving_path=None):
    losses = utils.AverageMeter()
    trainable_spectrogram.eval()
    X, X_hat = None, None
    with torch.no_grad():
        for x, _ in valid_sampler:
            x = x.to(device)
            x = x[:,:,::ds]

            with amp_autocast(enabled=use_amp and (device.type == "cuda")):
                X_hat = trainable_spectrogram(x)
                X = encoder(x)
                loss = torch.nn.functional.mse_loss(X_hat, X)

            if not torch.isfinite(loss):
                continue  # ignorer les batches de validation avec NaN
            losses.update(loss.item(), X.size(1))

        if stft_saving_path is not None and X is not None:
            save_spectrogram(X, stft_saving_path)
        if magssm_saving_path is not None and X_hat is not None:
            save_spectrogram(X_hat, magssm_saving_path)

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


def main():
    parser = argparse.ArgumentParser(description="Open trainable_spectrogram Distributed Trainer")

    # which target do we want to train?
    parser.add_argument("--target", type=str, default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset parameters

    parser.add_argument("--ds", type = int, default = 1, help="downsampling factor for the input data")

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
        default="open-trainable_spectrogram",
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
                        help="Number of frequencies in the trainable spectrogram. "
                             "Standard choice is to set it equal to the number of states.")

    parser.add_argument("--chunk-dur", type=float, default=6.0,
                        help="chunk duration in seconds. The input sequence will be split "
                             "into chunks for computation by the SSM module.")

    parser.add_argument("--mel", action="store_true", default=False,
                        help="If set, will initialize eigenvalue arguments of the A-matrix "
                             "on a log scale to enhance resolution in the lower frequency domain.")

    parser.add_argument("--bandwidth",
                        type=int, default=16000, help="maximum model bandwidth in herz")
    parser.add_argument("--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument("--nb-workers",
                        type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )

    parser.add_argument("--hidden_size_factors", type=int, nargs="+", default=None,
        help="Hidden size reduction factors for SEdge layers.")

    parser.add_argument("--output_size_factors", type=int, nargs="+", default=None,
        help="Output size increase factors for SEdge layers.")

    # Distributed Training Parameters
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use (default: nccl)")

    # Multi-node Parameters (pour torchrun --nnodes > 1)
    # Ces valeurs sont normalement passées via les variables d'environnement
    # MASTER_ADDR et MASTER_PORT par torchrun, mais on les expose aussi en arguments.
    parser.add_argument("--master-addr", type=str, default=None,
                        help="Adresse IP ou hostname du noeud master (rank 0). "
                             "Surcharge la variable d'environnement MASTER_ADDR.")
    parser.add_argument("--master-port", type=str, default="29500",
                        help="Port TCP du noeud master (défaut: 29500). "
                             "Surcharge la variable d'environnement MASTER_PORT.")

    # Misc Parameters
    parser.add_argument("--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument("--no-cuda",
                        action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--amp",
                        action="store_true", default=False,
                        help="Use automatic mixed precision (AMP) during training")

    args, _ = parser.parse_known_args()

    # ---------------------------------------------------------------------------
    # Initialisation du groupe de processus distribué (si lancé via torchrun)
    # ---------------------------------------------------------------------------
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        # Permettre de surcharger MASTER_ADDR/MASTER_PORT via arguments CLI
        # (utile pour les clusters sans scheduler qui injecte ces variables)
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
        # set_device est déjà fait en top-of-file, mais on le répète ici pour
        # sécuriser le cas où le script serait appelé sans la garde top-of-file.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        global_rank = 0
        local_rank  = 0
        world_size  = 1

    # Silence les rangs non-0 (tqdm, prints, etc.)
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

    # Seed (chaque rang a le même seed — le DistributedSampler gère le shuffle par rang)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------------------------------------------------------------
    # Chargement du dataset
    # Rang 0 en premier, puis barrier, puis les autres rangs.
    # Évite les race conditions si le dataset crée des fichiers cache.
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

    # Création du répertoire de sortie (rank 0 uniquement)
    target_path = Path(args.output)
    if global_rank == 0:
        target_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # DataLoaders avec DistributedSampler si DDP
    # ---------------------------------------------------------------------------
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
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate
    )

    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)

    # Freeze encoder (STFT classique — pas de paramètres à entraîner)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
        "nb_magssm_states": args.nb_magssm_states,
    }

    if global_rank == 0:
        with open(Path(target_path, "separator.json"), "w") as outfile:
            outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    # ---------------------------------------------------------------------------
    # Construction du modèle
    # ---------------------------------------------------------------------------
    if args.model:  # fine-tune depuis un modèle existant
        print_rank0(f"Fine-tuning model from {args.model}")
        trainable_spectrogram = utils_edge_var.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True, magssm=True
        )[args.target]
        trainable_spectrogram = trainable_spectrogram.to(device)

    else:
        chunk_duration_in_frames = int(args.chunk_dur * args.sample_rate)
        d_out = args.nb_magssm_states if args.d_out is None else args.d_out

        trainable_spectrogram = Trainable_spectrogram(
            nb_bins=args.nfft // 2 + 1,
            nb_channels=args.nb_channels,
            n_hop=args.nhop,
            dim_state=args.nb_magssm_states,
            encoder=encoder,
            device=device,
            chunk_duration=chunk_duration_in_frames,
            log_distributed_frequencies=args.mel,
        ).to(device)

        total_params = sum(p.numel() for p in trainable_spectrogram.parameters() if p.requires_grad)
        print_rank0(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.AdamW(
        trainable_spectrogram.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # Gradient scaler pour AMP (no-op si AMP désactivé)
    amp_scaler = amp_grad_scaler(enabled=args.amp and use_cuda)

    # ---------------------------------------------------------------------------
    # Reprise depuis checkpoint (AVANT le wrap DDP)
    # ---------------------------------------------------------------------------
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        trainable_spectrogram.load_state_dict(checkpoint["state_dict"], strict=False)
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

    # ---------------------------------------------------------------------------
    # Wrap DDP (après chargement du checkpoint pour éviter les conflits de clés)
    # ---------------------------------------------------------------------------
    if is_distributed:
        trainable_spectrogram = torch.nn.parallel.DistributedDataParallel(
            trainable_spectrogram,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Historique des valeurs propres — un enregistrement par époque
    lambda_log_path = Path(target_path, args.target + "_lambda.json")
    if lambda_log_path.exists():
        with open(lambda_log_path) as f:
            lambda_history = json.load(f)  # reprend si checkpoint
    else:
        lambda_history = []  # liste de dicts {epoch, stats_par_module}

    # ---------------------------------------------------------------------------
    # Boucle d'entraînement
    # ---------------------------------------------------------------------------
    stft_saving_path = Path(target_path, args.target + "_stft.png") if global_rank == 0 else None
    magssm_saving_path = Path(target_path, args.target + "_magssm.png") if global_rank == 0 else None

    for epoch in t:
        if is_distributed:
            train_sampler_ddp.set_epoch(epoch)

        t.set_description("Training epoch")
        end = time.time()

        train_loss = train(
            args, trainable_spectrogram, encoder, device, train_sampler, optimizer,
            is_distributed=is_distributed, scaler=amp_scaler, ds = args.ds
        )
        valid_loss = valid(
            args, trainable_spectrogram, encoder, device, valid_sampler,
            is_distributed=is_distributed, use_amp=args.amp, ds = args.ds,
            stft_saving_path=stft_saving_path, magssm_saving_path=magssm_saving_path
        )

        # Scheduler et early stopping sur tous les rangs (la loss de validation est synchronisée)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        # Sauvegarde uniquement sur le rank 0
        if global_rank == 0:
            if valid_loss == es.best:
                if stft_saving_path is not None and stft_saving_path.exists():
                    shutil.copyfile(stft_saving_path, Path(target_path, args.target + "_stft_best.png"))
                if magssm_saving_path is not None and magssm_saving_path.exists():
                    shutil.copyfile(magssm_saving_path, Path(target_path, args.target + "_magssm_best.png"))

            raw_state_dict = (
                trainable_spectrogram.module.state_dict()
                if is_distributed
                else trainable_spectrogram.state_dict()
            )
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

            # Sauvegarde du JSON de métriques
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

            # --- Enregistrement de la trajectoire des valeurs propres ---
            # On accède au module sous-jacent si DDP
            raw_model = (
                trainable_spectrogram.module
                if is_distributed
                else trainable_spectrogram
            )
            lambda_stats = collect_lambda_stats(raw_model)
            lambda_history.append({"epoch": epoch, "lambda": lambda_stats})
            with open(lambda_log_path, "w") as f:
                json.dump(lambda_history, f, indent=2)

            # Alerte console si des états deviennent instables
            for mod_name, s in lambda_stats.items():
                if s.get("n_nan_params", 0) > 0:
                    print(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: NaN dans Lambda ({s['n_nan_params']} params)")
                elif s.get("n_unstable", 0) > 0:
                    print(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: {s['n_unstable']} états instables "
                          f"(Re(Λ·Δ)>0), ld_re_max={s['ld_re_max']:.3e}")
                elif s.get("ld_re_max") is not None and s["ld_re_max"] > -1e-3:
                    print(f"[Lambda] ! EPOCH {epoch} — {mod_name}: ld_re_max={s['ld_re_max']:.3e} "
                          f"(Re(Λ·Δ) proche de 0)")

        train_times.append(time.time() - end)

        if stop:
            print_rank0("Apply Early Stopping")
            break

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
