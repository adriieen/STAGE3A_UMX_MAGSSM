import argparse
import torch
import torch.distributed as dist
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
import sedge_mask
import utils_edge_var


tqdm.monitor_interval = 0


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def train(args, unmix, encoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        Y_hat = unmix(x) # x is the waveform -- Mixture audio signal 
        Y = encoder(y)

        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
    return losses.avg


def valid(args, unmix, encoder, device, valid_sampler, is_distributed=False):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)

            Y_hat = unmix(x)
            Y = encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
            
    if is_distributed:
        # Sum validation loss and count across all processes
        loss_sum = torch.tensor(losses.sum, device=device)
        count_sum = torch.tensor(losses.count, device=device)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_sum, op=dist.ReduceOp.SUM)
        global_avg = (loss_sum / count_sum).item()
        return global_avg
    else:
        return losses.avg


def get_statistics(args, encoder, dataset):
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))

    return scaler.mean_, std


def main():
    parser = argparse.ArgumentParser(description="Open Unmix Distributed Trainer")

    # which target do we want to train?
    parser.add_argument("--target", type=str, default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset paramaters
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
        default="open-unmix",
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

    # Model Parameters

    parser.add_argument("--use_edge", 
        action="store_true",
        default = False,
        help = "Uses 'nb_layers' S-edge Layers for the main sequence to sequence" \
        "mapping from the separation module : otherwise, basic LSTM will be used."
    )

    parser.add_argument("--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )
    parser.add_argument("--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument("--nfft", type=int, default=4096, help="(STFT) fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="(STFT) hop size")

    parser.add_argument("--nb_magssm_states", type=int, default = 129, help= "Number of states in MAGSSM ")

    parser.add_argument("--d_out", type=int, default = None, 
                        help = "Number of frequencies in the trainable spectrogram." \
                        "Standard choice is to set it equal to the number of states, but it can be higher... ")
    
    parser.add_argument("--use_magssm", action="store_true", default = False, 
                        help = "If put as an argument, will use a MAGSSM that will proceed by chunks on the input " \
                        "not to overload the RAM")
    
    parser.add_argument("--chunk-dur", type=float, default = 6.0, 
                        help = "chunk duration in seconds. Only relevant if flag 'use_magssm' is set." \
                        "The input sequence will be split into chunks for computation by the Magssm with memory limit.")

    parser.add_argument("--mel", action="store_true", default = False,
                        help = "If put as an argument, will initialize the states of magssm along a MEL bankfliters")

    parser.add_argument("--hidden-size",
        type=int,
        default=512,
        help="hidden size parameter of bottleneck layers",
    )
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

    parser.add_argument("--nb_layers", type=int, default=3, help="Number of internal layers in the separation module")

    # Distributed Training Parameters
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use (default: nccl)")

    # Misc Parameters
    parser.add_argument("--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument("--no-cuda", 
                        action="store_true", default=False, help="disables CUDA training"
    )

    args, _ = parser.parse_known_args()

    # Initialize distributed process group if running under torchrun
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend=args.backend, init_method="env://")
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1

    # Silent other ranks
    args.quiet = args.quiet or (global_rank != 0)

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print_rank0("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        repo = Repo(repo_dir)
        commit = repo.head.commit.hexsha[:7]
    except Exception:
        commit = "unknown"

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        if global_rank == 0:
            train_dataset, valid_dataset, args = data.load_datasets(parser, args)
        dist.barrier()
        if global_rank != 0:
            train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    else:
        train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    args.sample_rate = train_dataset.sample_rate

    # create output dir if not exist
    target_path = Path(args.output)
    if global_rank == 0:
        target_path.mkdir(parents=True, exist_ok=True)

    # Set up Distributed Samplers
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
        valid_sampler = torch.utils.data.DataLoader(
            valid_dataset, batch_size=1, **dataloader_kwargs
        )

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
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
        "nb_magssm_states": args.nb_magssm_states
    }

    if global_rank == 0:
        with open(Path(target_path, "separator.json"), "w") as outfile:
            outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    max_bin = None
    scaler_mean = None
    scaler_std = None

    if args.model: # fine tune model
        print_rank0(f"Fine-tuning model from {args.model}")
        unmix = utils_edge_var.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True, magssm=True
        )[args.target]
        unmix = unmix.to(device)
    else:
        chunk_duration_in_frames = int(args.chunk_dur * args.sample_rate)
        d_out = args.nb_magssm_states if args.d_out is None else args.d_out

        unmix = sedge_mask.SedgeMask(
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_bins=args.nfft // 2 + 1,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            max_bin=max_bin,
            nb_layers=args.nb_layers,
            hidden_size_factors=args.hidden_size_factors,
            output_size_factors=args.output_size_factors,
            n_fft=args.nfft,
            n_hop=args.nhop,
            dim_state=args.nb_magssm_states,
            d_out=d_out,
            encoder=encoder,
            device=device,
            use_edge=args.use_edge,
            unidirectional=args.unidirectional,
            use_magssm=args.use_magssm,
            chunk_duration=chunk_duration_in_frames,
            mel=args.mel
        ).to(device)

        total_params = sum(p.numel() for p in unmix.parameters() if p.requires_grad)
        print_rank0(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # Resume training if checkpoint specified
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

    # Wrap model with DistributedDataParallel after potentially loading checkpoint
    if is_distributed:
        unmix = torch.nn.parallel.DistributedDataParallel(
            unmix,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    for epoch in t:
        if is_distributed:
            train_sampler_ddp.set_epoch(epoch)
        t.set_description("Training epoch")
        end = time.time()
        
        train_loss = train(args, unmix, encoder, device, train_sampler, optimizer)
        valid_loss = valid(args, unmix, encoder, device, valid_sampler, is_distributed)
        
        # Scheduler and early stopping run on all ranks since validation loss is synced
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        # Save checkpoint only on global rank 0
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

        train_times.append(time.time() - end)

        if stop:
            print_rank0("Apply Early Stopping")
            break

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
