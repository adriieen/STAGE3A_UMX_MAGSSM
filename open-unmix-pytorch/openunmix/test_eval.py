import torch
import sys
from path_config import setup_paths
setup_paths(add_openunmix_src=True)
import utils_edge_var

model_path = '/home/adubois/openunmix/OpenUnmix/outputs/500ep/magssm'
targets = ['vocals'] # Assuming vocals exists, adjust if needed

# Let's find what targets exist
import glob
import os
targets = [os.path.basename(f).replace('.json', '') for f in glob.glob(model_path + '/*.json') if 'separator' not in f]
print("Found targets:", targets)

separator = utils_edge_var.load_separator(
    model_str_or_path=model_path,
    targets=targets,
    niter=1,
    residual=True,
    device='cpu',
    pretrained=True,
    filterbank='torch',
    magssm=True
)

separator.freeze()
audio = torch.randn(1, 2, 44100 * 3) # 3 seconds of stereo audio

try:
    estimates = separator(audio)
    print("Success! Estimates shape:", estimates.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
