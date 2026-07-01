import sys
import os
from pathlib import Path
import torch
import numpy as np

# Setup paths
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(workspace_root / "open-unmix-pytorch" / "openunmix"))
sys.path.append(str(workspace_root / "SEdge" / "src"))

import data
import magssm_umx
import utils
import transforms
import model

def diagnose():
    print("=== STARTING DIAGNOSTIC SCRIPT ===")
    
    # 1. Check dataset path and contents
    dataset_path = Path("/Data/adrien.dubois/musdb18_ds3")
    print(f"Checking dataset path: {dataset_path}")
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset path '{dataset_path}' does not exist on this machine!")
        return
    else:
        file_count = len(list(dataset_path.glob("**/*.wav")))
        print(f"✓ Dataset path exists. Found {file_count} .wav files.")
        if file_count == 0:
            print("❌ WARNING: No wav files found in the dataset directory!")

    # 2. Check pre-trained checkpoints existence and values
    backbone_dir = Path("/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/UMX/eps0.1000_l1_4.0000_l2_1.7152")
    ssm_dir = Path("/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/fine_tuning_models/SSMs/eps0.1000_l1_4.0000_l2_1.7152")
    
    print("\n--- Checking Checkpoints ---")
    for name, p in [("Backbone", backbone_dir), ("SSM", ssm_dir)]:
        print(f"Checking {name} path: {p}")
        if not p.exists():
            print(f"❌ ERROR: {name} path '{p}' does not exist!")
            continue
        
        chkpnt_file = next(p.glob("vocals*.chkpnt"), None) or next(p.glob("vocals*.pth"), None)
        if not chkpnt_file:
            print(f"❌ ERROR: No checkpoint file found in {p}")
            continue
            
        print(f"✓ Found checkpoint: {chkpnt_file}")
        try:
            state = torch.load(chkpnt_file, map_location="cpu")
            state_dict = state.get("state_dict", state)
            
            # Check for NaNs/Infs in checkpoint weights
            nan_params = []
            inf_params = []
            for k, v in state_dict.items():
                if torch.isnan(v).any():
                    nan_params.append(k)
                if torch.isinf(v).any():
                    inf_params.append(k)
            
            if nan_params:
                print(f"❌ ERROR: Found NaN values in parameters: {nan_params}")
            if inf_params:
                print(f"❌ ERROR: Found Inf values in parameters: {inf_params}")
            if not nan_params and not inf_params:
                print(f"✓ No NaNs or Infs found in {name} checkpoint weights.")
        except Exception as e:
            print(f"❌ ERROR reading checkpoint {chkpnt_file}: {e}")

    # 3. Test dataset loading and iteration (no DDP, num_workers=0)
    print("\n--- Testing Dataloader (num_workers=0) ---")
    try:
        class DummyArgs:
            dataset = "musdb"
            root = str(dataset_path)
            target = "vocals"
            seq_dur = 6.0
            samples_per_track = 64
            source_augmentations = ["gain", "channelswap"]
            seed = 42
            is_wav = True
            
        class MockParser:
            def add_argument(self, *args, **kwargs):
                pass
            def parse_args(self, args=None):
                return DummyArgs()
                
        parser = MockParser()
        args = DummyArgs()
        train_dataset, valid_dataset, _ = data.load_datasets(parser, args)
        print(f"✓ Dataset loaded. Train tracks: {len(train_dataset.mus.tracks)}, Valid tracks: {len(valid_dataset.mus.tracks)}")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=0
        )
        
        print("Iterating over the first 3 batches...")
        for idx, (x, y) in enumerate(train_loader):
            print(f"  Batch {idx}: input shape {x.shape}, target shape {y.shape}")
            if torch.isnan(x).any():
                print(f"  ❌ ERROR: Input batch {idx} contains NaNs!")
            if torch.isnan(y).any():
                print(f"  ❌ ERROR: Target batch {idx} contains NaNs!")
            if idx >= 2:
                break
        print("✓ Dataloader iteration test complete.")
    except Exception as e:
        print(f"❌ ERROR during dataloader test: {e}")

    # 4. Test forward pass (no DDP)
    print("\n--- Testing Model Forward Pass ---")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load SSM parameters
        nfft, nhop, nb_magssm_states = 682, 34, 682
        d_out = nfft // 2 + 1
        
        stft, _ = transforms.make_filterbanks(
            n_fft=nfft, n_hop=nhop, sample_rate=14700
        )
        encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=False)).to(device)
        encoder.eval()
        
        unmix = magssm_umx.MagSSM_OpenUnmix(
            nb_bins=d_out,
            nb_channels=2,
            hidden_size=170,
            nb_layers=3,
            dim_state=nb_magssm_states,
            d_out=d_out,
            n_fft=nfft,
            n_hop=nhop,
            encoder=encoder,
            device=device,
            chunk_duration=int(6.0 * 14700),
            use_layernorm=True,
            og=True,
            eps_stability=0.0,
            dt_min=0.001,
            dt_max=0.1
        ).to(device)
        
        # Dummy batch
        dummy_input = torch.randn(2, 2, int(6.0 * 14700)).to(device)
        dummy_target = torch.randn(2, 2, int(6.0 * 14700)).to(device)
        
        with torch.no_grad():
            y_hat = unmix(dummy_input)
            loss = torch.nn.functional.mse_loss(y_hat, encoder(dummy_target))
            print(f"✓ Forward pass output shape: {y_hat.shape}")
            print(f"✓ Loss value: {loss.item()}")
            if torch.isnan(y_hat).any():
                print("❌ ERROR: Output contains NaNs!")
            if torch.isnan(loss):
                print("❌ ERROR: Loss is NaN!")
                
    except Exception as e:
        print(f"❌ ERROR during forward pass test: {e}")

if __name__ == "__main__":
    diagnose()
