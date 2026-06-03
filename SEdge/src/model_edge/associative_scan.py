import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from functools import partial
from functorch import vmap


from typing import Callable, overload, Any, List, Iterable, TypeVar, Tuple
from torch import Tensor


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
    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype)  # Cast to correct complex type
    # Static timesteps
    Bu_elements = cinput_sequence@B.T + B_bias.view(1,1,-1)    #B,T,C * C,H


    # Bu_elements = vmap(lambda u: B @ u)(cinput_sequence) + B_bias

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[1], 1)
    

    # _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))
    _, xs = vmap(lambda Bu : associative_scan(binary_operator, (Lambda_bars, Bu)))(Bu_elements)
    # _, xs = torch.vmap(lambda Bu : associative_scan(binary_operator, (Lambda_bars, Bu)))(Bu_elements)
    
    if complex_output:
        # out = vmap(lambda x: (C @ x))(xs) + C_bias
        out = xs@C.T + C_bias.view(1,1,-1)
    else:
        # out = (vmap(lambda x: (C @ x))(xs) + C_bias).real
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
    
    h = subsampling_factor
    _, c, _ = input_sequence.shape

    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype)  # Cast to correct complex type
    # Static timesteps
    Bu_elements = cinput_sequence@B.T + B_bias.view(1,1,-1)    #B,T,C *  C,H


    if last_state is not None : 
        Bu_elements[:, 0, :] += Lambda_bars * last_state
    


    # Bu_elements = vmap(lambda u: B @ u)(cinput_sequence) + B_bias

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[1], 1)

    # _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))
    _, xs = vmap(lambda Bu : associative_scan(binary_operator, (Lambda_bars, Bu)))(Bu_elements) #B,T,H
    # _, xs = torch.vmap(lambda Bu : associative_scan(binary_operator, (Lambda_bars, Bu)))(Bu_elements)
    
    
    # #downsampling here to limit nb of operation if we dont need full out -----  y_i for i ≠ k*hop_length
    # bc final goal is to downsample by factor hop_length.....

    last_state = xs[:,-1,:].to(torch.float32) # B, H
    
    xs_subsampled = xs[:,offset::h, :] #B, T/h, H
     

    if complex_output:
        # out = vmap(lambda x: (C @ x))(xs) + C_bias
        out = xs_subsampled@C.T + C_bias.view(1,1,-1) # B, T/h, d_out
    else:
        # out = (vmap(lambda x: (C @ x))(xs) + C_bias).real
        out = (xs_subsampled@C.T + C_bias.view(1,1,-1)).real
    
    return last_state, out


