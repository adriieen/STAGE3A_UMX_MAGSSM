"""Test dynamic phase correction for SSM decoder.

Run with:
    /users/eleves-a/2023/adrien.dubois/.conda/envs/umx310train/bin/python test_phase_correction.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from path_config import setup_paths
setup_paths()

import torch
import math
from model_edge.ssm import SSM


def seed_all(seed=42):
    torch.manual_seed(seed)


def test_B_matrix_unmodified():
    """Verify that B matrix remains standard orthogonal under --phase-correction."""
    seed_all(42)
    d_in, d_state, d_out = 4, 16, 1

    seed_all(42)
    ssm_pc = SSM(
        d_in=d_in, d_state=d_state, d_out=d_out,
        dt_min=0.005, dt_max=0.5, step_scale=1.0,
        B_C_init='orthogonal',
        structured_initialisation=True,
        init_with_positive_frequencies=False,
        domain="time",
        phase_correction=True
    )

    seed_all(42)
    ssm_no_pc = SSM(
        d_in=d_in, d_state=d_state, d_out=d_out,
        dt_min=0.005, dt_max=0.5, step_scale=1.0,
        B_C_init='orthogonal',
        structured_initialisation=True,
        init_with_positive_frequencies=False,
        domain="time",
        phase_correction=False
    )

    # B matrices should be identical (unmodified)
    B_diff = (ssm_pc.B.data - ssm_no_pc.B.data).abs().max().item()
    print(f"[B preservation] Absolute difference: {B_diff:.2e}")
    assert B_diff < 1e-7, f"B was unexpectedly modified: {B_diff}"

    print("✓ test_B_matrix_unmodified PASSED\n")


def test_forward_pass_dynamic_pc():
    """Verify that dynamic phase correction computes during forward pass and produces finite outputs."""
    seed_all(42)
    d_in, d_state, d_out = 4, 16, 1
    n_fft = 16

    ssm = SSM(
        d_in=d_in, d_state=d_state, d_out=d_out,
        dt_min=0.005, dt_max=0.5, step_scale=1.0,
        B_C_init='orthogonal',
        structured_initialisation=True,
        init_with_positive_frequencies=False,
        domain="time",
        phase_correction=True
    )

    x = torch.randn(2, n_fft, d_in)
    with torch.no_grad():
        y = ssm(x)

    print(f"[Dynamic Forward] Input shape: {x.shape}, Output shape: {y.shape}")
    assert y.shape == (2, n_fft, d_out), f"Unexpected output shape: {y.shape}"
    assert not torch.isnan(y).any(), "Output contains NaN"
    assert not torch.isinf(y).any(), "Output contains Inf"

    print("✓ test_forward_pass_dynamic_pc PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running dynamic phase correction unit tests (Fast CPU execution)")
    print("=" * 60 + "\n")

    test_B_matrix_unmodified()
    test_forward_pass_dynamic_pc()

    print("=" * 60)
    print("All unit tests PASSED ✓")
    print("=" * 60)
