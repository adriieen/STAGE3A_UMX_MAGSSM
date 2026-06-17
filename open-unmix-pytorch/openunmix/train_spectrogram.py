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


tqdm.monitor_interval = 0


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
                "ld_im_q1":  float(torch.quantile(ld_im, 0.25)) if not n_nan else None,
                "ld_im_q3":  float(torch.quantile(ld_im, 0.75)) if not n_nan else None,
                "ld_im_med":  float(torch.median(ld_im)) if not n_nan else None,
                # |Lambda_bar| = exp(Re(Λ·Δ))
                "lbar_mag_mean": float(mag.mean()) if not n_nan else None,
                "lbar_mag_max":  float(mag.max())  if not n_nan else None,
                # Indicateurs de danger
                "n_unstable":  int((ld_re > 0.0).sum())    if not n_nan else -1,
                "n_near_zero": int((ld_re > -1e-3).sum())  if not n_nan else -1,
                "n_nan_params": int(torch.isnan(L).sum()),
            }
    return stats

def save_spectrograms(true_spectrogram, predicted_spectrogram, save_path: str):
    """
    Saves the true and predicted spectrograms as a JSON file.
    Tensors have format (1, C, F, T) — batch=1, C=2 channels, F frequencies, T frames.
    The JSON structure is:
      {
        "true_spectrogram":      list[list[list[list[float]]]],  # shape (1, C, F, T)
        "predicted_spectrogram": list[list[list[list[float]]]],  # shape (1, C, F, T)
      }
    """
    true_np = true_spectrogram.detach().cpu().numpy()
    pred_np = predicted_spectrogram.detach().cpu().numpy()
    data = {
        "true_spectrogram": true_np.tolist(),
        "predicted_spectrogram": pred_np.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(data, f)

def L2_im_lambda(model, alpha, beta):
    try:
        from model_edge.ssm_bis import Progressive_SSM
    except ImportError:
        return 0.0

    """
    loss = alpha * 1/N sum [w_i**2]
    with w_i = ld_im - pi + beta if ld_im>pi and 0 otherwise,  ld_im = Im(Lambda * Delta) in rad/sample.

    NOTE: computed fully in PyTorch (no detach / no numpy) so that
    gradients flow back to Lambda and log_step, and the optimizer
    actually feels the regularization penalty.
    """

    base = model.module if hasattr(model, 'module') else model
    penalties = []

    for name, mod in base.named_modules():
        if not isinstance(mod, Progressive_SSM):
            continue

        # Keep full gradient graph 
        L = mod.Lambda.float()                             # [N, 2]
        log_step = mod.log_step.float()                    # [N]
        step = mod.step_scale * torch.exp(log_step)        # Delta [N]

        Lambda_c = torch.complex(L[:, 0], L[:, 1])
        LD = Lambda_c * step                               # Lambda * Delta [N]
        ld_im = LD.imag                                    # rad/sample [N]

        # Penalise im > pi  
        excess = torch.nn.functional.relu(ld_im - torch.pi)
        w = excess + beta * (excess > 0).float()

        penalties.append(w.pow(2).mean())

    if not penalties:
        return 0.0

    return alpha * torch.stack(penalties).mean()


def train(args, trainable_spectrogram, encoder, device, train_sampler, optimizer, scaler=None, ds = 1, alpha = 1e-2, beta = 0):
    losses = utils.AverageMeter()
    nan_batches = 0
    trainable_spectrogram.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, _ in pbar:   # (B, 2, L)
        pbar.set_description("Training batch")
        x = x.to(device)
        optimizer.zero_grad()
        x = x[:, : , ::ds]

        use_amp = (scaler is not None) and (device.type == "cuda")
        with amp_autocast(enabled=use_amp):
            X_hat = trainable_spectrogram(x) # x is the waveform -- Mixture audio signal 
            X = encoder(x)
            loss = torch.nn.functional.mse_loss(X_hat, X) + L2_im_lambda(trainable_spectrogram, alpha, beta)

        if not torch.isfinite(loss):
            nan_batches += 1
            print(f"[WARN] Loss non-finie ({loss.item():.4g}) sur ce batch — batch ignore.")
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
        print(f"[WARN] {nan_batches} batch(s) ignores (NaN/Inf) durant cette epoque.")
    return losses.avg if losses.count > 0 else float('nan')


def valid(args, trainable_spectrogram, encoder, device, valid_sampler, use_amp=False, ds=1, alpha = 1e-2, beta = 0, saving_path = None):
    losses = utils.AverageMeter()
    trainable_spectrogram.eval()
    X, X_hat = None, None
    with torch.no_grad():
        for x, _ in valid_sampler:
            x = x.to(device)
            x = x[:, : , ::ds]

            with amp_autocast(enabled=use_amp and (device.type == "cuda")):
                X_hat = trainable_spectrogram(x)
                X = encoder(x)      
                loss = torch.nn.functional.mse_loss(X_hat, X) + L2_im_lambda(trainable_spectrogram, alpha, beta)

            if not torch.isfinite(loss):
                continue
            losses.update(loss.item(), X.size(1))
        
        if saving_path is not None and X is not None and X_hat is not None:
            save_spectrograms(X, X_hat, saving_path)
        
        return losses.avg if losses.count > 0 else float('nan')



def main():
    parser = argparse.ArgumentParser(description="Open trainable_spectrogram Trainer")

    # which target do we want to train?
    parser.add_argument("--target", type=str, default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset paramaters

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
    parser.add_argument( "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument( "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")

    parser.add_argument("--alpha", type=float, default = 0, help = "factor that multiplies the L2 loss of the imaginary parts of the eigenvalues of the MAGSSM. No regularization per default")
    
    parser.add_argument("--beta", type=float, default = 1, help = "offset on the imaginary parts (frequencies) of the eigenvalues of the MAGSSM that are above pi if one wishes to have a stricter boundary.")

    parser.add_argument("--eps-stability", type=float, default=1e-3,
        help="Stability margin epsilon used in Progressive_SSM.forward() to clamp eigenvalue real parts "
             "away from 0. Smaller = tighter stability constraint. (default: 1e-3)")

    parser.add_argument("--dt-min", type=float, default=0.001,
        help="Minimum timescale (dt) for SSM log-step initialization. (default: 1e-3)")

    parser.add_argument("--dt-max", type=float, default=0.1,
        help="Maximum timescale (dt) for SSM log-step initialization. (default: 0.1)")

    # Model Parameters

    parser.add_argument("--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )

    parser.add_argument("--nfft", type=int, default=4096, help="(STFT) fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="(STFT) hop size")


    parser.add_argument("--nb_magssm_states", type=int, default = 129, help= "Number of states in MAGSSM ")
                                                    #2049 if we keep the upper config as default but we expect that less is in fact needed....


    parser.add_argument("--d_out", type=int, default = None, 
                        help = "Number of frequencies in the trainable spectrogram." \
                        "Standard choice is to set it equal to the number of states, but it can be higher... ")
    
    parser.add_argument("--chunk-dur", type=float, default = 6.0, # equiv to not progressive
                        help = "chunk duration in seconds. Only relevant if flag 'progressive' is set." \
                        "The input sequence will be split into chunks for computation by the SSM module")


    parser.add_argument("--mel", action="store_true", default = False,
                        help = "If put as an argument, will initialize the argument of the eigenvalues of the A-matrix " \
                        "according to a log scale to enhance resolution in the lower frequency domain")

    
    parser.add_argument("--bandwidth", 
                        type=int, default=16000, help="maximum model bandwidth in herz"
    )
    parser.add_argument("--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument("--nb-workers", 
                        type=int, default=0, help="Number of workers for dataloader."
    )
    parser.add_argument("--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )

    parser.add_argument("--hidden_size_factors", type = int, nargs = "+", default=None
        , help = "Give nb_layer factors. Hidden size in SEdge Layers will see their size reduced by the " \
        "factors the user precise in this field: ex for a desired decrease of 1;1/2;1/4" \
        "the user should write --hidden_....._factors 1 2 4 in the terminal")
    
    parser.add_argument("--output_size_factors", type = int, nargs = "+", default=None
        , help="Same thing that for the hidden size reduction but for the increase of the output size" \
        "in the output size of the SEdge layers. Note that last layer factor must be 1" \
        "ex for a desired increase in output size of 1/4 ; 1/2 ; 1 the user should write --output....._factors 4 2 1 in the terminal "\
        "A standard choice is to set output_sizes = hidden_sizes[::-1] to have a reasonable nb of parameters")
    


    # Misc Parameters
    parser.add_argument("--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument("--no-cuda", 
                        action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--amp",
                        action="store_true", default=False, help="Use automatic mixed precision (AMP) during training"
    )


    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        repo = Repo(repo_dir)
        commit = repo.head.commit.hexsha[:7]
    except Exception:
        commit = "unknown"

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    args.sample_rate = train_dataset.sample_rate // args.ds

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    stft, _ = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate 
    )


    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=args.nb_channels == 1)).to(device)

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate // args.ds,
        "nb_channels": args.nb_channels,
        "nb_magssm_states" : args.nb_magssm_states
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.model: # fine tune model
        print(f"Fine-tuning model from {args.model}")
        trainable_spectrogram = utils_spectrogram.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True
        )[args.target]
        trainable_spectrogram = trainable_spectrogram.to(device)

        # Récupérer les historiques du modèle source (tronqués à sa meilleure époque)
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
        
        chunk_duration_in_frames = int(args.chunk_dur * args.sample_rate) // args.ds
        d_out = args.nb_magssm_states if args.d_out is None else args.d_out

        scaler_mean, scaler_std = None, None
        
        trainable_spectrogram = Trainable_spectrogram(
            nb_bins = args.nfft // 2 + 1,
            nb_channels=args.nb_channels,
            n_hop = args.nhop,
            dim_state=args.nb_magssm_states,
            encoder = encoder,
            device = device,
            chunk_duration = chunk_duration_in_frames,
            log_distributed_frequencies= args.mel,
            eps_stability=args.eps_stability,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
        ).to(device)
    

        total_params = sum(p.numel() for p in trainable_spectrogram.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params}")


    optimizer = torch.optim.AdamW(trainable_spectrogram.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # Initialize gradient scaler for AMP
    scaler = amp_grad_scaler(enabled=args.amp and use_cuda)

    # if a checkpoint is specified: resume training
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        trainable_spectrogram.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
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
    # else start optimizer from scratch
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        # Si on fine-tune, initialiser les historiques depuis le modèle source
        if args.model:
            train_losses = ft_init_train_losses
            valid_losses = ft_init_valid_losses
            train_times  = ft_init_train_times
        else:
            train_losses = []
            valid_losses = []
            train_times  = []
        best_epoch = 0

    # Historique des valeurs propres — un enregistrement par époque
    lambda_log_path = Path(target_path, args.target + "_lambda.json")
    if lambda_log_path.exists():
        with open(lambda_log_path) as f:
            lambda_history = json.load(f)  # reprend si checkpoint
    elif args.model and not args.checkpoint:
        # Fine-tuning sans checkpoint : initialiser depuis le modèle source
        lambda_history = ft_init_lambda_history
    else:
        lambda_history = []  # liste de dicts {epoch, stats_par_module}

    saving_path = Path(target_path, args.target + "_spectrograms.json")

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        train_loss = train(args, trainable_spectrogram, encoder, device, train_sampler, optimizer, scaler=scaler, ds = args.ds, alpha=args.alpha, beta = args.beta)
        valid_loss = valid(args, trainable_spectrogram, encoder, device, valid_sampler, use_amp=args.amp, ds = args.ds, alpha=args.alpha, beta = args.beta, saving_path = saving_path)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch
            if saving_path.exists():
                shutil.copyfile(saving_path, Path(target_path, args.target + "_spectrograms_best.json"))

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": trainable_spectrogram.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target,
        )

        # save params
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
        lambda_stats = collect_lambda_stats(trainable_spectrogram)
        lambda_history.append({"epoch": epoch, "lambda": lambda_stats})
        with open(lambda_log_path, "w") as f:
            json.dump(lambda_history, f, indent=2)

        # Alerte console si des états deviennent instables
        for mod_name, s in lambda_stats.items():
            if s.get("n_nan_params", 0) > 0:
                print(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: NaN dans Lambda ({s['n_nan_params']} params)")
            elif s.get("n_unstable", 0) > 0:
                print(f"[Lambda] ⚠ EPOCH {epoch} — {mod_name}: {s['n_unstable']} états instables (Re(Λ·Δ)>0), ld_re_max={s['ld_re_max']:.3e}")
            elif s.get("ld_re_max") is not None and s["ld_re_max"] > -1e-3:
                print(f"[Lambda] ! EPOCH {epoch} — {mod_name}: ld_re_max={s['ld_re_max']:.3e} (Re(Λ·Δ) proche de 0)")


        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break



if __name__ == "__main__":
    main()
