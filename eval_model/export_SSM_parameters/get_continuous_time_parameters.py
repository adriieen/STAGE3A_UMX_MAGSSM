#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility to extract continuous-time SSM parameters from a trained PyTorch model checkpoint.
"""

import os
import json
import argparse
import numpy as np

# PyTorch is imported lazily to allow loading saved .npz files on machines without PyTorch.
torch = None

class SSMParameters:
    def __init__(self, Lambda, Delta, B, C, B_bias, C_bias, is_numpy=False):
        self.Lambda = Lambda
        self.Delta = Delta
        self.B = B
        self.C = C
        self.B_bias = B_bias
        self.C_bias = C_bias
        self._is_numpy = is_numpy

    @property
    def Lambda_d(self):
        """For backward compatibility: Lambda * Delta (Hadamard product)"""
        return self.Lambda * self.Delta

    def to_numpy(self):
        """Convert all parameters to NumPy arrays."""
        if self._is_numpy:
            return self
        
        def to_np(x):
            if hasattr(x, 'numpy'):
                return x.numpy()
            return x

        return SSMParameters(
            to_np(self.Lambda),
            to_np(self.Delta),
            to_np(self.B),
            to_np(self.C),
            to_np(self.B_bias),
            to_np(self.C_bias),
            is_numpy=True
        )

    def to_tensor(self):
        """Convert all parameters to PyTorch tensors."""
        if not self._is_numpy:
            return self
        
        global torch
        if torch is None:
            import torch
            
        return SSMParameters(
            torch.from_numpy(self.Lambda),
            torch.from_numpy(self.Delta),
            torch.from_numpy(self.B),
            torch.from_numpy(self.C),
            torch.from_numpy(self.B_bias),
            torch.from_numpy(self.C_bias),
            is_numpy=False
        )

    def save_npz(self, path):
        """Save parameters as a NumPy .npz file."""
        np_params = self.to_numpy()
        np.savez(
            path,
            Lambda=np_params.Lambda,
            Delta=np_params.Delta,
            B=np_params.B,
            C=np_params.C,
            B_bias=np_params.B_bias,
            C_bias=np_params.C_bias
        )
        print(f"Saved parameters to {path}")

    @classmethod
    def load_npz(cls, path):
        """Load parameters from a NumPy .npz file."""
        data = np.load(path)
        # Fallback with zeros if loading an older npz file that didn't have biases
        b_shape = data['B'].shape[0]
        c_shape = data['C'].shape[0]
        B_bias = data['B_bias'] if 'B_bias' in data.files else np.zeros(b_shape, dtype=np.complex64)
        C_bias = data['C_bias'] if 'C_bias' in data.files else np.zeros(c_shape, dtype=np.complex64)
        
        if 'Lambda' in data.files:
            Lambda_val = data['Lambda']
        else:
            # Reconstruct Lambda from older Lambda_d format
            Lambda_val = data['Lambda_d'] / data['Delta']
            
        return cls(
            Lambda_val,
            data['Delta'],
            data['B'],
            data['C'],
            B_bias,
            C_bias,
            is_numpy=True
        )

    def __repr__(self):
        type_str = "NumPy" if self._is_numpy else "PyTorch"
        return (f"SSMParameters ({type_str})\n"
                f"  Lambda   : {self.Lambda.shape}\n"
                f"  Delta    : {self.Delta.shape}\n"
                f"  B        : {self.B.shape}\n"
                f"  C        : {self.C.shape}\n"
                f"  B_bias   : {self.B_bias.shape}\n"
                f"  C_bias   : {self.C_bias.shape}")


def as_complex(t, dtype=None):
    global torch
    if torch is None:
        import torch
    if dtype is None:
        dtype = torch.complex64
        
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


def get_parameters(pth_path_or_dir):
    """
    Reads a .pth file (or loads a .npz file) and returns continuous-time parameters as an SSMParameters object.
    
    Parameters:
    -----------
    pth_path_or_dir : str
        Path to the .pth checkpoint file, its containing directory, or a saved .npz file.
        
    Returns:
    --------
    SSMParameters
        An object with attributes Lambda_d, Delta, B, C, B_bias, C_bias.
    """
    # If the user passed a .npz file, load it directly using numpy (no PyTorch needed)
    if isinstance(pth_path_or_dir, str) and pth_path_or_dir.endswith('.npz'):
        return SSMParameters.load_npz(pth_path_or_dir)

    # Otherwise we need PyTorch to load the .pth checkpoint
    global torch
    if torch is None:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required to parse .pth checkpoints. "
                "Please install PyTorch or load a pre-saved .npz file instead."
            )

    # 1. Resolve file paths
    if os.path.isdir(pth_path_or_dir):
        # Look for .pth and .json files in the directory
        pth_files = [f for f in os.listdir(pth_path_or_dir) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in directory {pth_path_or_dir}")
        pth_name = 'vocals.pth' if 'vocals.pth' in pth_files else pth_files[0]
        pth_path = os.path.join(pth_path_or_dir, pth_name)
        
        # Look for matching json
        base_name = os.path.splitext(pth_name)[0]
        json_path = os.path.join(pth_path_or_dir, f"{base_name}.json")
        if not os.path.exists(json_path):
            json_files = [f for f in os.listdir(pth_path_or_dir) if f.endswith('.json') and f != 'separator.json']
            if json_files:
                json_path = os.path.join(pth_path_or_dir, json_files[0])
            else:
                json_path = None
    else:
        pth_path = pth_path_or_dir
        # Try to find json next to the pth file
        base_path = os.path.splitext(pth_path)[0]
        json_path = f"{base_path}.json"
        if not os.path.exists(json_path):
            # Try any json in the same directory
            dir_name = os.path.dirname(pth_path)
            json_files = [f for f in os.listdir(dir_name) if f.endswith('.json') and f != 'separator.json'] if dir_name else []
            if json_files:
                json_path = os.path.join(dir_name, json_files[0])
            else:
                json_path = None

    print(f"Loading weights from: {pth_path}")
    state_dict = torch.load(pth_path, map_location='cpu')

    # Load configuration if json exists
    config = {}
    if json_path and os.path.exists(json_path):
        print(f"Loading configuration from: {json_path}")
        with open(json_path, 'r') as f:
            config = json.load(f)
    
    args = config.get('args', {})

    # Extract parameters
    log_step = state_dict['magssm_encoder.mimo.seq.log_step']
    Lambda = state_dict['magssm_encoder.mimo.seq.Lambda']
    B = state_dict['magssm_encoder.mimo.seq.B']
    C = state_dict['magssm_encoder.mimo.seq.C']
    B_bias = state_dict['magssm_encoder.mimo.seq.B_bias']
    C_bias = state_dict['magssm_encoder.mimo.seq.C_bias']

    # Step scale (defaults to 1.0)
    step_scale = 1.0

    # Calculate Delta (step)
    Delta = step_scale * torch.exp(log_step)

    # Stability mode
    ensure_stability = args.get('ensure_stability', 'sigmoid_interval')
    print(f"Stability enforcement mode: {ensure_stability}")
    
    if ensure_stability == 'sigmoid_interval':
        # Retrieve bounds from state_dict buffers if present, otherwise fallback to args config
        re_lower = state_dict.get('magssm_encoder.mimo.seq.re_lower')
        if re_lower is None:
            re_lower = args.get('re_lower')
        else:
            re_lower = re_lower.item()

        re_upper = state_dict.get('magssm_encoder.mimo.seq.re_upper')
        if re_upper is None:
            re_upper = args.get('re_upper')
        else:
            re_upper = re_upper.item()

        sigmoid_scale = args.get('sigmoid_scale', 1.0)

        if re_lower is None or re_upper is None:
            raise ValueError("re_lower and re_upper must be provided in config or checkpoint for sigmoid_interval stability")

        print(f"Sigmoid interval parameters: re_lower={re_lower}, re_upper={re_upper}, sigmoid_scale={sigmoid_scale}")

        # Compute Re(lambda * Delta)
        width = re_upper - re_lower
        real_part_d = re_lower + width * torch.sigmoid(sigmoid_scale * Lambda[:, 0])
        # Compute physical continuous-time Lambda (Re = real_part_d / Delta, Im = Lambda[:, 1])
        real_part = real_part_d / Delta
        imag_part = Lambda[:, 1]
        lambda_c = torch.complex(real_part, imag_part)
        
    elif ensure_stability == 'abs':
        eps_stability = args.get('eps_stability', 0.0) / torch.min(Delta)
        # Apply abs stability
        real_part = -torch.abs(Lambda[:, 0]).clamp(min=eps_stability)
        imag_part = Lambda[:, 1]
        lambda_c = torch.complex(real_part, imag_part)
        
    elif ensure_stability == 'relu':
        eps_stability = args.get('eps_stability', 0.0) / torch.min(Delta)
        # Apply relu stability
        real_part = -torch.relu(- (Lambda[:, 0] + eps_stability)) - eps_stability
        imag_part = Lambda[:, 1]
        lambda_c = torch.complex(real_part, imag_part)
        
    else:
        # Default fallback: compute lambda_c without modification
        lambda_c = as_complex(Lambda)

    # Convert B, C, B_bias, C_bias to complex
    B_c = as_complex(B)
    C_c = as_complex(C)
    B_bias_c = as_complex(B_bias)
    C_bias_c = as_complex(C_bias)
    
    return SSMParameters(lambda_c, Delta, B_c, C_c, B_bias_c, C_bias_c, is_numpy=False)


# Legacy alias
get_continuous_time_parameters = get_parameters


def main():
    parser = argparse.ArgumentParser(description="Extract continuous-time parameters from UMX-MAGSSM checkpoint")
    parser.add_argument("path", type=str, nargs="?", 
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Path to the checkpoint (.pth) file, its directory, or a saved .npz file")
    parser.add_argument("--save-npz", type=str, default=None,
                        help="If specified, saves the parameters as a NumPy .npz file at the given path")
    args = parser.parse_args()

    try:
        params = get_parameters(args.path)
        
        # Convert to numpy for printing and saving
        np_params = params.to_numpy()
        
        print("\n=== Parameter Info ===")
        print(f"Lambda shape : {np_params.Lambda.shape} (complex)")
        print(f"Delta shape  : {np_params.Delta.shape}")
        print(f"B shape      : {np_params.B.shape} (complex)")
        print(f"C shape      : {np_params.C.shape} (complex)")
        print(f"B_bias shape : {np_params.B_bias.shape} (complex)")
        print(f"C_bias shape : {np_params.C_bias.shape} (complex)")
        
        print("\n=== Stats ===")
        print(f"Delta: min={np_params.Delta.min():.6e}, max={np_params.Delta.max():.6e}, mean={np_params.Delta.mean():.6e}")
        print(f"Re(Lambda): min={np_params.Lambda.real.min():.6e}, max={np_params.Lambda.real.max():.6e}, mean={np_params.Lambda.real.mean():.6e}")
        print(f"Im(Lambda): min={np_params.Lambda.imag.min():.6e}, max={np_params.Lambda.imag.max():.6e}, mean={np_params.Lambda.imag.mean():.6e}")
        
        if args.save_npz:
            np_params.save_npz(args.save_npz)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
