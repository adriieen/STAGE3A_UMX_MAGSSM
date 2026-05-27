import math
import torch
import numpy as np
# from .jax_compat import variance_scaling, lecun_normal, uniform
import scipy.linalg

from typing import Optional, Sequence
from typing_extensions import Literal
# Initialization Functions

def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A



def make_linear_eigenvalues(N, symmetric = True):
    """ Create a S4D-Lin vector.
        Args:
            N (int32): state size
        Returns:
            N  complex eigenvalues
    """
    if symmetric:
        Lambda = -0.5 + 1j * np.arange(-N//2, N//2)
    else:
        # Lambda = -0.5 + 1j * np.arange(N)
        Lambda = -0.5 + 1j * np.linspace(0,N//2,N)

    lambda_real = np.expand_dims(Lambda.real, axis=1)
    lambda_imag = np.expand_dims(Lambda.imag, axis=1)
    Lambda = np.concatenate((lambda_real, lambda_imag), axis=1)
    Lambda = torch.tensor(Lambda, dtype=torch.float)
    return Lambda

import numpy as np
import torch

def make_spectrograms_log_eigenvalues(N, samplerate, mel = True):


    log_sigma = torch.nn.Parameter(torch.log(torch.ones(N) * 2))

    if not mel:  # standard Fourier basis
        nyq_f =  samplerate / 2 
        freqs = np.linspace(0, nyq_f, N)
        freqs = np.maximum(freqs, 1e-7)
        log_omega = torch.log(torch.tensor(freqs * 2 * np.pi, dtype=torch.float32))
    
    else:  # more states resonates on lower frequencies (Mel scale)
        nyq_freq = samplerate / 2
        max_mel = 1127 * np.log(1 + nyq_freq / 700)
        mel_scale = np.linspace(0, max_mel, N)
        freqs = 700 * (np.exp(mel_scale / 1127) - 1)
        freqs = np.maximum(freqs, 1e-7)
        log_omega = torch.log(torch.tensor(freqs * 2 * np.pi, dtype=torch.float32))

    return log_sigma, log_omega

def make_spectrograms_eigenvalues_correction(N, mel = True):

    sigma = - torch.ones(N) / 2

    if not mel:
        omega = torch.linspace(0,torch.pi, N)
    
    else:
        mel_scale = torch.linspace(0, torch.log(1+torch.pi), N)
        omega = torch.tensor(torch.exp(mel_scale) - 1, dtype=torch.float32)

    sigma = sigma.unsqueeze(1)
    omega = omega.unsqueeze(1)

    Lambda = torch.cat((sigma, omega), dim=1)
    return Lambda


        




def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def make_Normal_S(N):
    nhippo = make_HiPPO(N)
    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]
    return S


def make_Normal_HiPPO(N, B=1):
    """Create a normal approximation to HiPPO-LegS matrix.
    For HiPPO matrix A, A=S+pqT is normal plus low-rank for
    a certain normal matrix S and low rank terms p and q.
    We are going to approximate the HiPPO matrix with the normal matrix S.
    Note we use original numpy instead of jax.numpy first to use the
    onp.linalg.eig function. This is because Jax's linalg.eig function does not run
    on GPU for non-symmetric matrices. This creates tracing issues.
    So we instead use onp.linalg eig and then cast to a jax array
    (since we only have to do this once in the beginning to initialize).
    Args:
        N (int32): state size
        B (int32): diagonal blocks
    Returns:
        Lambda (complex64): eigenvalues of S (N,)
        V      (complex64): eigenvectors of S (N,N)
    """

    assert N % B == 0, "N must divide blocks"
    S = (make_Normal_S(N // B),) * B
    S = scipy.linalg.block_diag(*S)

    # Diagonalize S to V \Lambda V^*
    Lambda, V = np.linalg.eig(S)

    # Convert to jax array
    return torch.tensor(Lambda), torch.tensor(V)


def uniform(shape, dtype=torch.float, minval=0., maxval=1.0, device=None):
    src = torch.rand(shape, dtype=dtype, device=device)
    if minval == 0 and maxval == 1.:
        return src
    else:
        return (src * (maxval - minval)) + minval
    
def _complex_uniform(shape: Sequence[int],
                     dtype, device=None) -> torch.Tensor:
    """
    Sample uniform random values within a disk on the complex plane,
    with zero mean and unit variance.
    """
    r = torch.sqrt(2 * torch.rand(shape, dtype=dtype, device=device))
    theta = 2 * torch.pi * torch.rand(shape, dtype=dtype, device=device)
    return r * torch.exp(1j * theta)



def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return uniform(shape, minval=np.log(dt_min), maxval=np.log(dt_max))
        # return torch.rand(shape) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)

    return init


def init_log_steps(H, dt_min, dt_max):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(shape=(1,))
        log_steps.append(log_step)

    return torch.tensor(log_steps)


def _compute_fans(shape, fan_in_axes=None):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        if fan_in_axes is not None:
            # Compute fan-in using user-specified fan-in axes.
            fan_in = np.prod([shape[i] for i in fan_in_axes])
            fan_out = np.prod([s for i, s in enumerate(shape)
                              if i not in fan_in_axes])
        else:
            # If no axes specified, assume convolution kernels (2D, 3D, or more.)
            # kernel_shape: (..., input_depth, depth)
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out



def _complex_truncated_normal(upper: float,
                              shape: Sequence[int],
                              dtype, device=None) -> torch.Tensor:
    """
    Sample random values from a centered normal distribution on the complex plane,
    whose modulus is truncated to `upper`, and the variance before the truncation
    is one.
    """
    real_dtype = torch.tensor(0, dtype=dtype).real.dtype
    t = ((1 - torch.exp(torch.tensor(-(upper ** 2), dtype=dtype, device=device)))
         * torch.rand(shape, dtype=real_dtype, device=device).type(dtype))
    r = torch.sqrt(-torch.log(1 - t))
    theta = 2 * torch.pi * torch.rand(shape, dtype=real_dtype, device=device).type(dtype)
    return r * torch.exp(1j * theta)


def _truncated_normal(lower, upper, shape, dtype=torch.float):
    if shape is None:
        shape = torch.broadcast_shapes(np.shape(lower), np.shape(upper))

    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)

    # a<u<b
    u = uniform(shape, dtype, minval=a, maxval=b)
    out = sqrt2 * torch.erfinv(u)
    # Clamp the value to the open interval (lower, upper) to make sure that
    # rounding (or if we chose `a` for `u`) doesn't push us outside of the range.
    with torch.no_grad():
        return torch.clip(
            out,
            torch.nextafter(torch.tensor(lower), torch.tensor(np.inf, dtype=dtype)),
            torch.nextafter(torch.tensor(upper), torch.tensor(-np.inf, dtype=dtype)))


def variance_scaling(scale: float,
                     mode: Literal["fan_in", "fan_out", "fan_avg"] = 'fan_in',
                     distribution: Literal["truncated_normal", "normal", "uniform"] = 'truncated_normal',
                     fan_in_axes: Optional[Sequence[int]] = None,
                     dtype=torch.float):
    def init(shape: Sequence[float],
             dtype=dtype,
             device=None):
        fan_in, fan_out = _compute_fans(shape, fan_in_axes)
       
        if mode == 'fan_in':
            denom = max(1, fan_in)
        elif mode ==  'fan_out':
            denom = max(1, fan_out)
        elif mode == 'fan_avg':
            denom = max(1, (fan_in + fan_out) / 2)
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")

        variance = scale/denom
      
        if distribution == 'normal':
            return torch.normal(0, np.sqrt(variance), shape, dtype=dtype, device=device)
        elif distribution == 'uniform':
            if dtype.is_complex:
                return _complex_uniform(shape, dtype=dtype, device=device) * np.sqrt(variance)
            else:
                return uniform(shape, dtype=dtype, device=device, minval=-1, maxval=1.0) * np.sqrt(3 * variance)
        elif distribution == 'truncated_normal':
            if dtype.is_complex:
                stddev = np.sqrt(variance) * 0.95311164380491208
                return _complex_truncated_normal(2, shape, dtype=dtype, device=device) * stddev
            else:
                stddev = np.sqrt(variance) * 0.87962566103423978
                return _truncated_normal(-2., 2., shape, dtype=dtype) * stddev
        else:
            raise ValueError(f"invalid distribution for variance scaling initializer: {distribution}")
        
    return init

def lecun_normal(fan_in_axes=None, dtype=torch.float):
    """Builds a Lecun normal initializer.

    A `Lecun normal initializer`_ is a specialization of
    :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
    ``mode="fan_in"``, and ``distribution="truncated_normal"``.

    Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

    Returns:
    An initializer.

    Example:

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.lecun_normal()
    >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
    Array([[ 0.46700746,  0.8414632 ,  0.8518669 ],
         [-0.61677957, -0.67402434,  0.09683388]], dtype=float32)

    .. _Lecun normal initializer: https://arxiv.org/abs/1706.02515
    """
    return variance_scaling(1.0, "fan_in", "truncated_normal", fan_in_axes=fan_in_axes, dtype=dtype)



def trunc_standard_normal(shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        C = lecun_normal()(shape=(1, P, 2))
        Cs.append(C)
    Cs = np.array(Cs)
    return torch.tensor(Cs)[:, 0]


def init_columnwise_B(shape, dtype):
    """Initialize B matrix in columnwise fashion.
    We will sample each column of B from a lecun_normal distribution.
    This gives a different fan-in size then if we sample the entire
    matrix B at once. We found this approach to be helpful for PathX
    It appears to be related to the point in
    https://arxiv.org/abs/2206.12037 regarding the initialization of
    the C matrix in S4, so potentially more important for the
    C initialization than for B.
     Args:
         key: jax random key
         shape (tuple): desired shape, either of length 3, (P,H,_), or
                      of length 2 (N,H) depending on if the function is called
                      from the low-rank factorization initialization or a dense
                      initialization
     Returns:
         sampled B matrix (float32), either of shape (H,P) or
          shape (H,P,2) (for complex parameterization)
    """
    shape = shape[:2] + ((2,) if len(shape) == 3 else ())
    lecun = variance_scaling(0.5 if len(shape) == 3 else 1.0, fan_in_axes=(0,))
    return lecun(shape, dtype)



def init_rowwise_C(shape, dtype):
    """Initialize C matrix in rowwise fashion. Analogous to init_columnwise_B function above.
    We will sample each row of C from a lecun_normal distribution.
    This gives a different fan-in size then if we sample the entire
    matrix B at once. We found this approach to be helpful for PathX.
    It appears to be related to the point in
    https://arxiv.org/abs/2206.12037 regarding the initialization of
    the C matrix in S4.
     Args:
         shape (tuple): desired shape, of length 3, (H,P,_)
     Returns:
         sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    shape = shape[:2] + ((2,) if len(shape) == 3 else ())
    lecun = variance_scaling(0.5, fan_in_axes=(0,))
    return lecun(shape, dtype)


def S5_init(in_dim, out_dim, hidden_dim):
    # General conversion
    # d_model = H == hidden_dim (, out_dim [as no difference exists in S5])
    # ssm_size_base = ssm_size = P == hidden_dim

    hidden_dim = 2*hidden_dim
    # required from S5
    if hidden_dim % 8 == 0:
        blocks = hidden_dim//8
    elif hidden_dim % 4 == 0:
        print("blocks = hidden_dim//4 was used")
        blocks = hidden_dim//4
    else:
        raise ValueError("Hidden dimension must be divisible by 4 or 8")
    conj_sym = True	


    # partial coppy from S5
    block_size = int(hidden_dim / blocks)              # train.py line 33
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)   # train.py line 76
    if conj_sym:
        block_size = block_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T                                         # train.py line 84

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    V = scipy.linalg.block_diag(*([V] * blocks))
    Vinv = scipy.linalg.block_diag(*([Vc] * blocks))

    Lambda = np.stack([Lambda.real, Lambda.imag], axis=-1)  
    Lambda = torch.tensor(Lambda, dtype=torch.float)        

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))               # train.py line 94

    # C init = trunc_standard_normal
    C = lecun_normal()(shape=(out_dim, hidden_dim, 2))      # ssm.py line 175
    C = C[..., 0] + 1j * C[..., 1]                          # ssm_init.py line 164
    C = C @ V                                               # ssm_init.py line 165
    C = np.stack([C.real, C.imag], axis=-1)                 # ssm_init.py line 167-168
    C = torch.tensor(C, dtype=torch.float)                  

    # B init
    B = lecun_normal()(shape=(hidden_dim, in_dim)).numpy()  # ssm_init.py line 127
    B = Vinv @ B                                            # ssm_init.py line 128
    B = np.stack([B.real, B.imag], axis=-1)                 # ssm_init.py line 129-131 
    B = torch.tensor(B, dtype=torch.float)

    

    print("B.shape={}".format(B.shape))
    print("C.shape={}".format(C.shape))


    return Lambda, B, C 