import argparse
import functools
import json
import os
import sys
from typing import Optional, Union
from pathlib import Path

import torch
import torch.distributed as dist

# Set CUDA device ASAP, before importing anything else (like torchaudio, which might initialize CUDA)
is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if is_distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
else:
    local_rank = 0

import musdb
import museval
import tqdm
import utils
import utils_edge_var

def separate_and_evaluate(
    track: musdb.MultiTrack,
    targets: list,
    model_str_or_path: str,
    niter: int,
    output_dir: str,
    eval_dir: str,
    residual: bool,
    mus,
    aggregate_dict: dict = None,
    device: Union[str, torch.device] = "cpu",
    wiener_win_len: Optional[int] = None,
    filterbank="torch",
    use_edge = False
) -> str:

    if use_edge: 
        separator = utils_edge_var.load_separator(
            model_str_or_path=model_str_or_path,
            targets=targets,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True,
            filterbank=filterbank,
        )
    else:
        separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            targets=targets,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True,
            filterbank=filterbank,
        )

    separator.freeze()
    separator.to(device)

    audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)

    if use_edge:
        audio = utils_edge_var.preprocess(audio, track.rate, separator.sample_rate)
    else:   
        audio = utils.preprocess(audio, track.rate, separator.sample_rate)

    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)

    for key in estimates:
        estimates[key] = estimates[key][0].cpu().detach().numpy().T

    estimates['accompaniment'] = audio.cpu().squeeze().detach().numpy().T - estimates['vocals']

    if output_dir:
        mus.save_estimates(estimates, track, output_dir)

    scores = museval.eval_mus_track(track, estimates, output_dir=eval_dir)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUSDB18 DDP Parallel Evaluation", add_help=False)

    parser.add_argument("--targets",
        nargs="+",
        default=["vocals", "drums", "bass", "other"],
        type=str,
        help="provide targets to be processed. \
              If none, all available targets will be computed",
    )

    parser.add_argument("--model",
        default="umxl",
        type=str,
        help="path to mode base directory of pretrained models",
    )

    parser.add_argument("--outdir",
        type=str,
        help="Results path where audio evaluation results are stored",
    )

    parser.add_argument("--evaldir", type=str, help="Results path for museval estimates")

    parser.add_argument("--root", type=str, help="Path to MUSDB18")

    parser.add_argument("--subset", type=str, default="test", help="MUSDB subset (`train`/`test`)")

    parser.add_argument("--cores", type=int, default=1)

    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA inference")

    parser.add_argument("--is-wav",
        action="store_true",
        default=False,
        help="flags wav version of the dataset",
    )

    parser.add_argument("--niter",
        type=int,
        default=1,
        help="number of iterations for refining results.",
    )

    parser.add_argument("--wiener-win-len",
        type=int,
        default=300,
        help="Number of frames on which to apply filtering independently",
    )

    parser.add_argument( "--residual",
        type=str,
        default=None,
        help="if provided, build a source with given name for the mix minus all estimated targets",
    )

    parser.add_argument( "--aggregate",
        type=str,
        default=None,
        help="if provided, must be a string containing a valid expression for "
        "a dictionary, with keys as output target names, and values "
        "a list of targets that are used to build it. For instance: "
        '\'{"vocals":["vocals"], "accompaniment":["drums",'
        '"bass","other"]}\'',
    )

    parser.add_argument("--use_edge", 
        action="store_true",
        default = False,
        help = "Uses 'nb_layers' S-edge Layers for the main sequence to sequence" \
        "mapping from the separation module : otherwise, basic LSTM will be used."
    )

    parser.add_argument("--backend",
        type=str,
        default="gloo",
        help="Distributed backend to use (gloo or nccl)",
    )

    args = parser.parse_args()

    # Initialize Distributed process group
    if is_distributed:
        dist.init_process_group(backend=args.backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Distributed initialized with backend {args.backend}. World size: {world_size}")

    # Load dataset
    mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets=args.subset,
        is_wav=args.is_wav,
    )
    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)

    # Shard tracks list
    all_tracks = mus.tracks
    local_tracks = [track for i, track in enumerate(all_tracks) if i % world_size == rank]

    if rank == 0:
        print(f"Total tracks: {len(all_tracks)}. Shard sizes across ranks:")
    
    print(f"Rank {rank} processing {len(local_tracks)} tracks...")

    # Run evaluations for local tracks
    local_results = []
    pbar = tqdm.tqdm(local_tracks, desc=f"Rank {rank}", disable=(rank != 0))
    for track in pbar:
        scores = separate_and_evaluate(
            track,
            targets=args.targets,
            model_str_or_path=args.model,
            niter=args.niter,
            residual=args.residual,
            mus=mus,
            aggregate_dict=aggregate_dict,
            output_dir=args.outdir,
            eval_dir=args.evaldir,
            device=device,
            wiener_win_len=args.wiener_win_len,
            use_edge=args.use_edge
        )
        local_results.append((track.name, scores))

    # Synchronize and gather all results
    if is_distributed:
        dist.barrier()
        all_results = [None] * world_size
        dist.all_gather_object(all_results, local_results)
    else:
        all_results = [local_results]

    # Rank 0 aggregates results and saves them
    if rank == 0:
        results = museval.EvalStore()
        
        # Build map track_name -> scores
        name_to_scores = {}
        for rank_res in all_results:
            if rank_res is not None:
                for track_name, scores in rank_res:
                    name_to_scores[track_name] = scores
                    
        # Add to EvalStore in the original order of mus.tracks
        for track in all_tracks:
            if track.name in name_to_scores:
                results.add_track(name_to_scores[track.name])
            else:
                print(f"[WARNING] Missing scores for track {track.name}!")

        print("\n=== FINAL RESULTS ===")
        print(results)
        
        # Save results
        method = museval.MethodStore()
        method.add_evalstore(results, args.model)
        method.save(args.model + ".pandas")
        print(f"Saved evaluation results to {args.model}.pandas")

    if is_distributed:
        dist.destroy_process_group()
