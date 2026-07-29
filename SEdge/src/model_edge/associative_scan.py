import sys
import math
import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from functools import partial
try:
    from functorch import vmap
except ImportError:
    vmap = getattr(torch, "vmap", None)

vmap_fn = vmap if sys.version_info < (3, 10) else torch.vmap


from typing import Callable, overload, Any, List, Iterable, TypeVar, Tuple
from torch import Tensor

from path_config import amp_autocast


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

@overload
def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> List[T]: ...


@overload
def safe_map(f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]) -> List[T]: ...


@overload
def safe_map(f: Callable[[T1, T2, T3], T], __arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3]) -> List[T]: ...


@overload
def safe_map(f: Callable[..., T], __arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], *args) -> List[T]: ...


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))

def combine(tree, operator, a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat

def _interleave(a, b, axis: int):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    b_trunc = (a.shape[axis] == b.shape[axis] + 1)
    if b_trunc:
        pad = [0, 0] * b.ndim
        pad[(b.ndim-axis-1)*2+1] = 1 # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis+1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis+1)
    if b_trunc:
        # TODO: find torch alternative for slice_along axis for torch.jit.script to work
        interleaved = torch.ops.aten.slice(interleaved, axis, 0, b.shape[axis]+a.shape[axis]-1)
    return interleaved

def _scan(tree, operator, elems, axis: int):
    """Perform scan on `elems`."""
    num_elems = elems[0].shape[axis]

    if num_elems < 2:
        return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(tree, operator,
      [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
      [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems])

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(tree, operator, reduced_elems, axis)

    if num_elems % 2 == 0:
        even_elems = combine(tree, operator,
            [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])
    else:
        even_elems = combine(tree, operator,
            odd_elems,
            [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
      torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
      if result.shape.numel() > 0 and elem.shape[axis] > 0 else
      result if result.shape.numel() > 0 else
      torch.ops.aten.slice(elem, axis, 0, 1)  # Jax allows/ignores concat with 0-dim, Pytorch does not
      for (elem, result) in zip(elems, even_elems)]

    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))


@torch.jit.script
def binary_operator(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    # return A_j * A_i, A_j * b_i + b_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)





# Pytorch impl. of jax.lax.associative_scan
def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool =False):
    # if not callable(operator):
    #     raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    assert axis >= 0 or axis < elems_flat[0].ndim, "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    scans = _scan(tree, operator, elems_flat, axis)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(scans, tree)

# Todo: Remove this function
# Apply only the associative scan inside the SSM layer -> more general 



def apply_ssm(Lambda_bars: torch.Tensor, B, B_bias, C, C_bias, input_sequence, complex_output):

    with amp_autocast(enabled=False):
        batch_size = input_sequence.shape[0]

        cinput_sequence = input_sequence.type(
            Lambda_bars.dtype)  # Cast to correct complex type
        # Static timesteps
        Bu_elements = cinput_sequence@B.T + B_bias.view(1,1,-1)    #B,T,C * C,H

        if Lambda_bars.ndim == 1:  # Repeat for associative_scan
            Lambda_bars = Lambda_bars.tile(input_sequence.shape[1], 1)
        
        Lambda_bars_expanded = Lambda_bars.unsqueeze(1).expand(-1, batch_size, -1)
        Bu_elements_permuted = Bu_elements.permute(1, 0, 2)

        _, xs_permuted = associative_scan(binary_operator, (Lambda_bars_expanded, Bu_elements_permuted), axis=0)
        xs = xs_permuted.permute(1, 0, 2)

        if complex_output:
            out = xs@C.T + C_bias.view(1,1,-1)
        else:
            out = (xs@C.T + C_bias.view(1,1,-1)).real
        return out




def apply_ssm_progressive(
        Lambda_bars: torch.Tensor, 
        B, 
        B_bias, 
        C, 
        C_bias, 
        input_sequence, 
        complex_output, 
        last_state = None,
        subsampling_factor = 1,
        offset = 0):

    with amp_autocast(enabled=False):
        h = subsampling_factor
        batch_size = input_sequence.shape[0]

        cinput_sequence = input_sequence.type(
            Lambda_bars.dtype)  # Cast to correct complex type
        # Static timesteps
        Bu_elements = cinput_sequence@B.T + B_bias.view(1,1,-1)    #B,T,C *  C,H

        if last_state is not None:
            first_step = Bu_elements[:, 0, :] + Lambda_bars * last_state
            if Bu_elements.shape[1] > 1:
                Bu_elements = torch.cat([first_step.unsqueeze(1), Bu_elements[:, 1:, :]], dim=1)
            else:
                Bu_elements = first_step.unsqueeze(1)

        if Lambda_bars.ndim == 1:  # Repeat for associative_scan
            Lambda_bars = Lambda_bars.tile(input_sequence.shape[1], 1)

        # Avoid vmap over batch dimension by permuting Bu_elements to (Time, Batch, States) 
        # and expanding Lambda_bars to match. This enables training with AMP and reduces memory usage.
        Lambda_bars_expanded = Lambda_bars.unsqueeze(1).expand(-1, batch_size, -1)
        Bu_elements_permuted = Bu_elements.permute(1, 0, 2)

        _, xs_permuted = associative_scan(binary_operator, (Lambda_bars_expanded, Bu_elements_permuted), axis=0)
        xs = xs_permuted.permute(1, 0, 2) # (Batch, Time, States)

        last_state = xs[:,-1,:] # B, H
        
        xs_subsampled = xs[:,offset::h, :] #B, T/h, H

    if complex_output:
        out = xs_subsampled@C.T + C_bias.view(1,1,-1) # B, T/h, d_out
    else:
        out = (xs_subsampled@C.T + C_bias.view(1,1,-1)).real
    
    return last_state, out




def apply_ssm_ft(
    Lambda_bars: torch.Tensor, 
    B_bar: torch.Tensor, 
    B_bias_bar: torch.Tensor, 
    C: torch.Tensor, 
    C_bias: torch.Tensor, 
    input_sequence: torch.Tensor, 
    complex_output: bool, 
    subsampling_factor: int = 1
):
    """
    Computes the SSM output using Fast Fourier Transform (FFT) convolution.
    
    Args:
        Lambda_bars: Discretized state transition matrix exp(Lambda * dt), shape (d_state,)
        B_bar: Discretized input matrix, shape (d_state, d_in)
        B_bias_bar: Discretized input bias, shape (d_state,)
        C: Output matrix, shape (d_out, d_state)
        C_bias: Output bias, shape (d_out,)
        input_sequence: Input tensor, shape (Batch, Time, d_in)
        complex_output: Whether output should remain complex or take real part
        subsampling_factor: Downsampling hop factor h
    """
    with amp_autocast(enabled=False):
        batch_size, L, d_in = input_sequence.shape
        d_state = Lambda_bars.shape[0]
        d_out = C.shape[0]

        cinput_sequence = input_sequence.type(Lambda_bars.dtype)

        # 1. Compute state powers Lambda_bars^t for t = 0 ... L-1
        log_Lambda = torch.log(Lambda_bars)  # (d_state,)
        t_steps = torch.arange(L, device=input_sequence.device, dtype=torch.float32)  # (L,)
        Lambda_powers = torch.exp(t_steps.unsqueeze(1) * log_Lambda.unsqueeze(0))  # (L, d_state)

        # 2. Compute SSM impulse response kernel K_t = sum_p C_{o, p} * Lambda_powers_{t, p} * B_{p, i}
        W = C.unsqueeze(2) * B_bar.unsqueeze(0)  # (d_out, d_state, d_in)
        W_perm = W.permute(1, 0, 2).reshape(d_state, d_out * d_in)  # (d_state, d_out * d_in)
        
        K_flat = Lambda_powers @ W_perm  # (L, d_out * d_in)
        K = K_flat.view(L, d_out, d_in)  # (L, d_out, d_in)

        # 3. FFT Convolution: pad to N_fft >= 2 * L
        N_fft = 2 ** math.ceil(math.log2(2 * L))

        u_fft = torch.fft.fft(cinput_sequence, n=N_fft, dim=1)  # (B, N_fft, d_in)
        K_fft = torch.fft.fft(K, n=N_fft, dim=0)  # (N_fft, d_out, d_in)

        # Frequency domain multiplication & summation over d_in
        Y_fft = torch.einsum('bfi, foi -> bfo', u_fft, K_fft)  # (B, N_fft, d_out)

        # Inverse FFT to return to time domain
        y_conv = torch.fft.ifft(Y_fft, dim=1)[:, :L, :]  # Truncate to original length L

        # Add input bias response if present
        if B_bias_bar is not None and torch.any(B_bias_bar != 0):
            W_bias = C * B_bias_bar.unsqueeze(0)  # (d_out, d_state)
            bias_response = Lambda_powers @ W_bias.T  # (L, d_out)
            y_conv = y_conv + bias_response.unsqueeze(0)

        # Downsample along time dimension if subsampling_factor > 1
        xs_subsampled = y_conv[:, ::subsampling_factor, :]

        # Add output bias
        out = xs_subsampled + C_bias.view(1, 1, -1)

        if not complex_output:
            out = out.real

        return out


