import musdb
import museval
import torch
import tqdm
from typing import Optional, Union

import sys
sys.path.append('/home/adubois/openunmix/OpenUnmix/open-unmix-pytorch/openunmix')
import utils
import utils1D
import utils_edge_var


import subprocess
from torch.profiler import profile, ProfilerActivity, record_function
from types import SimpleNamespace


models = {
    "edge" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/sedge",
    "1D" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/1d",
    "edge-var" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/sedge_var"
}

args = {
    "root" : "/dev/shm/musdb18/musdb18_wav",
    "targets" : "vocals",
    "niter" : 0,
    "outdir" : "/dev/null" ,
    "evaldir" : "/dev/null",
    "is_wav" : True,
    "subset" : "test",

}
args = SimpleNamespace(**args)


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
) -> str:

    # --- Sélection du loader (inchangé) ---
    if model_str_or_path == models["edge"] or model_str_or_path == models["edge-var"]:
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
    elif model_str_or_path == "/home/adubois/openunmix/OpenUnmix/outputs/500ep/1d":
         separator = utils1D.load_separator(
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

    # Préparation de l'audio
    audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)
    audio = utils.preprocess(audio, track.rate, separator.sample_rate)

    activities = [ProfilerActivity.CPU]



    with profile(
        activities=activities,
        with_flops=True,           # Option pour compter les FLOPs
        record_shapes=True,        # Optionnel : aide à débugger les dimensions
        profile_memory=False       # On se concentre sur les FLOPs
    ) as prof:
        with record_function("model_inference"):
            # Exécution de la séparation
            estimates = separator(audio)


    stats = prof.key_averages()
    
    # Calcul des FLOPs totaux (somme de toutes les opérations tracées)
    total_flops = sum(getattr(event, 'flops', 0) for event in stats)
    
    print("-" * 30)
    print(f"Total FLOPs (estimés) : {total_flops:.2e}") # Format scientifique
    
    # Affiche le top 10 des fonctions les plus lourdes en FLOPs
    print(prof.key_averages().table(sort_by="flops", row_limit=10))
    print("-" * 30)


if __name__ == "__main__":

    device = "cpu"

    for model in models.keys():

        mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets=args.subset,
        is_wav=args.is_wav,
        )   
        
        for track in tqdm.tqdm(mus.tracks):
            print(f"Flops for model {model} on track {track.name} : ")
            scores = separate_and_evaluate(
                track,
                targets=args.targets,
                model_str_or_path=models[model],
                niter=args.niter,
                residual=None,
                mus=mus,
                aggregate_dict=None,
                output_dir=args.outdir,
                eval_dir=args.evaldir,
                device=device,
            )
            break






