import json
import shutil
from pathlib import Path
import torch

def fix_checkpoint(dir_path: Path, target: str = "vocals"):
    pth_path = dir_path / f"{target}.pth"
    chkpnt_path = dir_path / f"{target}.chkpnt"
    json_path = dir_path / f"{target}.json"

    if not pth_path.exists():
        print(f"Error: {pth_path} does not exist. Cannot restore checkpoint.")
        return

    print("=== Step 1: Backup current files ===")
    if chkpnt_path.exists():
        shutil.copy(chkpnt_path, chkpnt_path.with_suffix(".chkpnt.bak"))
        print(f"Backed up corrupted checkpoint to {chkpnt_path.with_suffix('.chkpnt.bak')}")
    if json_path.exists():
        shutil.copy(json_path, json_path.with_suffix(".json.bak"))
        print(f"Backed up JSON to {json_path.with_suffix('.json.bak')}")

    print("\n=== Step 2: Load details from JSON ===")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    best_epoch = data.get("best_epoch")
    best_loss = data.get("best_loss")
    print(f"Best epoch found in JSON: {best_epoch} with loss: {best_loss}")

    # The best epoch index is 0-indexed. Since best_epoch is 107 (the 108th epoch),
    # the weights in .pth correspond to the state after completing epoch 107.
    # Therefore, we want to resume training starting at epoch 108.
    resume_epoch = best_epoch + 1
    print(f"We will roll back and resume training at epoch: {resume_epoch}")

    print("\n=== Step 3: Reconstruct .chkpnt from .pth ===")
    state_dict = torch.load(pth_path, map_location="cpu")
    checkpoint = {
        "epoch": resume_epoch,
        "state_dict": state_dict,
        "best_loss": best_loss,
        "optimizer": None,  # Optimizer will be reinitialized
        "scheduler": None,  # Scheduler will be reinitialized
    }
    torch.save(checkpoint, chkpnt_path)
    print(f"Successfully wrote clean checkpoint to {chkpnt_path}")

    print("\n=== Step 4: Truncate training history in JSON ===")
    data["epochs_trained"] = resume_epoch
    for key in ["train_loss_history", "valid_loss_history", "train_time_history"]:
        if key in data:
            data[key] = data[key][:resume_epoch]
            print(f"Truncated {key} to {len(data[key])} elements.")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)
    print(f"Successfully updated {json_path}")
    print("\nFinish! You can now run your resume script.")

if __name__ == "__main__":
    target_dir = Path("/users/eleves-a/2023/adrien.dubois/stage/STAGE3A_UMX_MAGSSM/outputs/umx_training/seed=40/eps0.5000_l1_15.0000_l2_4.0000")
    fix_checkpoint(target_dir)
