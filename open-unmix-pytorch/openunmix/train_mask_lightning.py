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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer


import data
import model
import utils
import transforms
import sedge_mask
import utils_edge_var


tqdm.monitor_interval = 0


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
                "ld_re_med": float(torch.median(ld_re)) if not n_nan else None,
                # Im(Lambda * Delta) — fréquence en rad/sample
                "ld_im_mean": float(ld_im.mean()) if not n_nan else None,
                "ld_im_min":  float(ld_im.min())  if not n_nan else None,
                "ld_im_max":  float(ld_im.max())  if not n_nan else None,
                "ld_im_med": float(torch.median(ld_im)) if not n_nan else None,
                # |Lambda_bar| = exp(Re(Λ·Δ))
                "lbar_mag_mean": float(mag.mean()) if not n_nan else None,
                "lbar_mag_max":  float(mag.max())  if not n_nan else None,
                # Indicateurs de danger
                "n_unstable":  int((ld_re > 0.0).sum())    if not n_nan else -1,
                "n_nan_params": int(torch.isnan(L).sum()),
            }
    return stats



class UnmixLitWrapper(pl.LightningModule):
    def __init__(self, unmix, encoder, args):
        super().__init__()
        self.unmix = unmix
        self._encoder = encoder
        self.args = args

        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad = False
    def forward(self, x):

        return self.unmix(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        Y_hat = self.unmix(x)
        Y = self._encoder(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        Y_hat = self.unmix(x)
        Y = self._encoder(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.unmix.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.args.lr_decay_gamma,
            patience=self.args.lr_decay_patience,
            cooldown=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class JsonLogCallback(pl.Callback):
    def __init__(self, target_path, args):
        super().__init__()
        self.target_path = target_path
        self.args = args
        self.train_losses = []
        self.valid_losses = []
        self.train_times = []
        self.lambda_history = []  # trajectoire des valeurs propres (Lambda * Delta)
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.epoch_start_time = 0
        # Si on reprend l'entraînement depuis un checkpoint, on recharge l'ancien JSON
        if args.checkpoint:
            json_path = Path(args.checkpoint).expanduser() / f"{args.target}.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    data = json.load(f)
                    self.train_losses = data.get("train_loss_history", [])
                    self.valid_losses = data.get("valid_loss_history", [])
                    self.train_times = data.get("train_time_history", [])
                    self.best_loss = data.get("best_loss", float('inf'))
                    self.best_epoch = data.get("best_epoch", 0)
            lambda_path = Path(args.checkpoint).expanduser() / "lambda_history.json"
            if lambda_path.exists():
                with open(lambda_path, "r") as f:
                    self.lambda_history = json.load(f)
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        # Récupère la train loss moyenne de l'epoch (Lightning ajoute _epoch si on_step=True)
        train_loss = trainer.callback_metrics.get("train_loss") or trainer.callback_metrics.get("train_loss_epoch")
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        
        self.train_times.append(time.time() - self.epoch_start_time)
        self._save_json(trainer, pl_module)
    def on_validation_epoch_end(self, trainer, pl_module):
        # Ignore la validation de test (sanity check) avant le début de l'entraînement
        if trainer.sanity_checking:
            return
        # Récupère la valid loss
        valid_loss = trainer.callback_metrics.get("val_loss")
        is_best = False
        if valid_loss is not None:
            val_loss_val = valid_loss.item()
            self.valid_losses.append(val_loss_val)
            
            if val_loss_val < self.best_loss:
                self.best_loss = val_loss_val
                self.best_epoch = trainer.current_epoch
                is_best = True
        self._save_json(trainer, pl_module)

        # Sauvegarde au format classique "pur" de l'ancien code (.chkpnt)
        if trainer.is_global_zero and valid_loss is not None:
            import utils
            checkpoint_dict = {
                "epoch": trainer.current_epoch + 1,
                "state_dict": pl_module.unmix.state_dict(),
                "best_loss": self.best_loss,
                "optimizer": {}, # Vide, Lightning gère ça dans son propre .ckpt
                "scheduler": {},
            }
            utils.save_checkpoint(
                checkpoint_dict,
                is_best=is_best,
                path=self.target_path,
                target=self.args.target,
            )

    def _save_json(self, trainer, pl_module):
        # Sauvegarde le JSON uniquement sur le GPU 0 (pour éviter que les 2 GPUs écrivent en même temps)
        # On vérifie que les listes ont la même taille et qu'elles ne sont pas vides
        if trainer.is_global_zero and len(self.train_losses) == len(self.valid_losses) and len(self.train_losses) > 0:
            # On récupère le nombre de bad epochs depuis le vrai callback de Lightning
            num_bad_epochs = 0
            from pytorch_lightning.callbacks import EarlyStopping
            for cb in trainer.callbacks:
                if isinstance(cb, EarlyStopping):
                    num_bad_epochs = getattr(cb, 'wait_count', 0)
                    break

            params = {
                "epochs_trained": trainer.current_epoch + 1,
                "args": vars(self.args),
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "train_loss_history": self.train_losses,
                "valid_loss_history": self.valid_losses,
                "train_time_history": self.train_times,
                "num_bad_epochs": num_bad_epochs,
            }
            with open(self.target_path / f"{self.args.target}.json", "w") as outfile:
                json.dump(params, outfile, indent=4, sort_keys=True)

            # Trajectoire des valeurs propres (Lambda * Delta) — collecte via pl_module.unmix
            lambda_stats = collect_lambda_stats(pl_module.unmix)
            self.lambda_history.append({"epoch": trainer.current_epoch, "lambda": lambda_stats})
            with open(self.target_path / "lambda_history.json", "w") as f:
                json.dump(self.lambda_history, f, indent=2)


def get_statistics(args, encoder, dataset):
    encoder = copy.deepcopy(encoder).to("cpu")  # CPU explicite → pas de conflit GPU
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
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)
        scaler.partial_fit(np.squeeze(X))

    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

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
                                                    #2049 if we keep the upper config as default but we expect that less is in fact needed....


    parser.add_argument("--d_out", type=int, default = None, 
                        help = "Number of frequencies in the trainable spectrogram." \
                        "Standard choice is to set it equal to the number of states, but it can be higher... ")
    
    
    parser.add_argument("--chunk-dur", type=float, default = 6.0, # equiv to not progressive
                        help = "chunk duration in seconds. Only relevant if flag 'progressive' is set." \
                        "The input sequence will be split into chunks for computation by the Magssm with memory limit.")


    parser.add_argument("--mel", action="store_true", default = False,
                        help = "If put as an argument, will initialize the states of magssm along a log_distributed_frequenciesbankfliters")


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

    # repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # repo = Repo(repo_dir)
    # commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    args.sample_rate = train_dataset.sample_rate

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

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
        "nb_magssm_states" : args.nb_magssm_states
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    if args.checkpoint or args.model or args.debug:
        scaler_mean = None
        scaler_std = None
    else:
        # scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)
        scaler_mean, scaler_std = None, None
    max_bin = None

    if args.model: # fine tune model
        print(f"Fine-tuning model from {args.model}")
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
            nb_layers = args.nb_layers,
            hidden_size_factors = args.hidden_size_factors,
            output_size_factors = args.output_size_factors,
            n_fft = args.nfft,
            n_hop = args.nhop,
            dim_state=args.nb_magssm_states,
            d_out = d_out,
            encoder  = encoder,
            device = device,
            use_edge = args.use_edge,
            unidirectional = args.unidirectional,
            # progressive = args.progressive,
            chunk_duration = chunk_duration_in_frames,
            log_distributed_frequencies= args.mel
            ).to(device)

        total_params = sum(p.numel() for p in unmix.parameters() if p.requires_grad)
        print(f"Total number of parameters: {total_params}")


        #TODO 
        # print expected memory usage for magssm step ~ Batch * chunk_duration * sample_rate * nb_of_states * 2(complex)
        # 


        # input_tensor = torch.rand((16,2,2049,255), dtype=torch.float32).to(device)
        # torch.onnx.export(
        #     unmix,
        #     (input_tensor,),
        #     "UMXEdge.onnx",
        #     input_names=["input"]
        # )


    lit_model = UnmixLitWrapper(unmix, encoder, args)
    # JSON Callback 
    json_callback = JsonLogCallback(target_path, args)
    # Saving model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=target_path,
        filename=f'{args.target}-best',
        save_top_k=1,
        mode='min',
    )
    # Early Stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[0, 1],
        strategy="ddp",
        precision=16 if args.amp else 32,
        callbacks=[checkpoint_callback, early_stop_callback, json_callback],
    )
    ckpt_path = args.checkpoint if args.checkpoint else None
    # Training
    trainer.fit(
        model=lit_model, 
        train_dataloaders=train_sampler, 
        val_dataloaders=valid_sampler,
        ckpt_path=ckpt_path
    )


if __name__ == "__main__":
    main()
