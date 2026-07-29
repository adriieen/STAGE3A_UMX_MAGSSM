from typing import Optional

import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from .associative_scan import apply_ssm, apply_ssm_progressive, apply_ssm_ft
from .init import make_linear_eigenvalues, init_log_steps, S5_init, make_structured_eigenvalues
import math


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt

def discretize_zoh(Lambda, B, B_bias, Delta, bias):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B      (complex64): input matrix + bias                (P, H + 1)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H + 1)
    """
    if bias:
        B_concat = torch.cat((B, B_bias.unsqueeze(1)), dim=-1)
    else:
        B_concat = B
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = ((Lambda_bar - 1)/Lambda)[..., None] * B_concat
    return Lambda_bar, B_bar

class SSM(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 dt_min: float,
                 dt_max: float,
                 step_scale: float = 1.0,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=False,
                 B_C_init='orthogonal',
                 C_C_init=None,
                 ensure_stability='abs',
                 symmetric=False,
                 subsampling_factor = 1,
                structured_initialisation = False,
                init_with_positive_frequencies = True,
                domain = "frequency",
                synthesis_window = None,
                target_tau = None,
                effective_samplerate = None,
                phase_correction = False,
                 ): 
        self.phase_correction = phase_correction
        """The Modified S5 SSM
        Args:
            d_in        (int32):     Number of features of input
            d_state     (int32):     state size
            d_out       (int32):     Number of output features
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        self.symmetric = symmetric
        self.target_tau = target_tau
        self.effective_samplerate = effective_samplerate

        self.log_step = torch.nn.Parameter(init_log_steps(d_state, dt_min, dt_max))

        
        # lambdaInit  (float32): Initial diagonal state matrix       (P,2)
        step = step_scale * torch.exp(self.log_step)

        if structured_initialisation:
            if domain == "frequency":
                N_bins = d_out
            elif domain == "time":
                N_bins = d_in
            else:
                raise NotImplementedError(f"domain {domain} not implemented")
            lambda_unscaled = make_structured_eigenvalues(d_state, N_bins, positive_frequencies=init_with_positive_frequencies, domain = domain,
            target_tau=self.target_tau, effective_samplerate=self.effective_samplerate)
            # print(lambda_unscaled.shape)
            
            lambda_scaled = torch.cat(((lambda_unscaled[:,0])[:,None], (lambda_unscaled[:,1]/step)[:,None]), dim=1)
            
            self.Lambda = torch.nn.Parameter(lambda_scaled)
        
        else: #standard linear initialisation
            self.Lambda = torch.nn.Parameter(make_linear_eigenvalues(d_state, symmetric=self.symmetric, positive_frequencies=init_with_positive_frequencies))
        
        print("Re_lambda_min / max", (self.Lambda.data[:,0]*step).min().item(), (self.Lambda.data[:,0]*step).max().item())
        if structured_initialisation:
            print("Im_lambda_min / max", lambda_unscaled.data[:,1].min().item(), lambda_unscaled.data[:,1].max().item())
        
        self.discretize = discretize_zoh

        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.step_scale = step_scale
        self.ensure_stability = ensure_stability
        self.subsampling_factor = subsampling_factor
        self.domain = domain

        if self.input_bias:
            if bias_init == 'zero':
                self.B_bias = torch.nn.Parameter(
                    torch.zeros(d_state, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.B_bias = torch.nn.Parameter(
                    torch.rand(d_state, 2, dtype=torch.float))
        else:
            self.B_bias = torch.nn.Parameter(torch.zeros(
                d_state, 2, dtype=torch.float), requires_grad=False)

        if self.output_bias:
            if bias_init == 'zero':
                self.C_bias = torch.nn.Parameter(
                    torch.zeros(d_out, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.C_bias = torch.nn.Parameter(
                    torch.rand(d_out, 2, dtype=torch.float))
        else:
            self.C_bias = torch.nn.Parameter(torch.zeros(
                d_out, 2, dtype=torch.float), requires_grad=False)
        if B_C_init == 'S5':
            lamb, B, C = S5_init(d_in, d_out, d_state)
            self.Lambda.data = lamb
            self.B = torch.nn.Parameter(B,requires_grad=True)
            self.C = torch.nn.Parameter(2*C,requires_grad=True)
            self.B_bias = torch.nn.Parameter(self.B_bias.data[:self.Lambda.shape[0],...],requires_grad=True)
            self.log_step = torch.nn.Parameter(init_log_steps(self.Lambda.shape[0], dt_min, dt_max))

            print('S5 init')
            print('A', self.Lambda.shape)
            print('B', self.B.shape)
            print('C', self.C.shape)
            print('B_bias', self.B_bias.shape)
            print('C_bias', self.C_bias.shape)


        elif B_C_init == 'orthogonal' or B_C_init is None:
            gain = np.sqrt(4/12)
            B_r = torch.empty(d_state, d_in)
            B_i = torch.empty(d_state, d_in)
            if d_in == 1:
                B_r = torch.nn.init.normal_(B_r, std=gain)
                B_i = torch.nn.init.normal_(B_i, std=gain)
            else:
                B_r = torch.nn.init.orthogonal_(B_r.T, gain=gain).T
                B_i = torch.nn.init.orthogonal_(B_i.T, gain=gain).T

            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))

            if C_C_init == "istft" and d_out == d_in:
                # C_{n, m} = \frac{1}{N_{\text{fft}}} \exp\left(j \frac{2\pi n m}{N_{\text{fft}}}\right)
                n_grid = np.arange(d_out)[:, None]
                m_grid = np.arange(d_state)[None, :]
                C_init = (1.0 / d_out) * np.exp(1j * 2 * np.pi * n_grid * (m_grid % d_out) / d_out)
                C_r = torch.tensor(C_init.real, dtype=torch.float)
                C_i = torch.tensor(C_init.imag, dtype=torch.float)

            else:
                C_r = torch.empty(d_out, d_state)
                C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
                C_i = torch.empty(d_out, d_state)
                C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_uniform_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_uniform_(B_i,  nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_uniform_(C_r,  nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_uniform_(C_i,  nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_normal_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_normal_(B_i,  nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_normal_(C_r,  nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_normal_(C_i,  nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_uniform_(
                B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_uniform_(
                B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_uniform_(
                C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_uniform_(
                C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_normal_(
                B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_normal_(
                B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_normal_(
                C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_normal_(
                C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))


    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C.shape[-2]))

    def forward_rnn(self, signal, prev_state):
        Lambda_c = as_complex(self.Lambda)
        if self.ensure_stability == 'relu':
            Lambda_c = torch.complex(-F.relu(-Lambda_c.real), Lambda_c.imag)
        elif self.ensure_stability == 'abs':
            Lambda_c = torch.complex(-torch.abs(Lambda_c.real), Lambda_c.imag)
        else:
            raise NotImplementedError(
                'Only relu and abs stability are implemented')

        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        cinput_sequence = signal.type(C_c.dtype)

        step = self.step_scale * torch.exp(self.log_step)
        Lambda_bar, B_bars = self.discretize(
            Lambda_c, B_c, B_bias_c, step, self.input_bias)

        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bar[:, 0])

        Bu = B_bar @ cinput_sequence + B_bias_bar
        x = Lambda_bar * prev_state + Bu
        y = C_c @ x + C_bias_c

        if self.complex_output:
            y_out = y
        else:
            y_out = y.real
        return y_out, x

    def forward(self, signal):
        with torch.no_grad():
            if self.ensure_stability == 'relu':
                self.Lambda.data[:, 0] = -F.relu(-self.Lambda.data[:, 0])
            elif self.ensure_stability == 'abs':
                self.Lambda.data[:, 0] = -torch.abs(self.Lambda.data[:, 0])

        Lambda = as_complex(self.Lambda)
        step = self.step_scale * torch.exp(self.log_step)

        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        Lambda_bars, B_bars = self.discretize(
            Lambda, B_c, B_bias_c, step, self.input_bias)
        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bars[:, 0])

        full_x = apply_ssm(Lambda_bars, B_bar, B_bias_bar, C_c, C_bias_c, signal, self.complex_output) # B,L,d_out
        x_downsampled = full_x[:,::self.subsampling_factor,:] #B,L/n_hop,d_out
        return x_downsampled




class Progressive_SSM(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 dt_min: float,
                 dt_max: float,
                 step_scale: float = 1.0,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=False,
                 og = False,
                 B_C_init= None,
                 C_C_init= None,
                 ensure_stability='abs',
                 symmetric=False,
                 chunk_duration = 264600,
                 subsampling_factor = 1,
                 log_distributed_frequencies= False,
                 samplerate = 44100.0,
                 eps_stability: float = 1e-3,
                 re_lower: float = None,
                 re_upper: float = None,
                 sigmoid_scale: float = 1.0,
                 structured_initialisation = False,
                 init_with_positive_frequencies = True,
                 domain = "frequency",
                 target_tau = None,
                 effective_samplerate = None
                 ): 
        """The Modified S5 SSM (Progressive Version)"""
        super().__init__()
        self.symmetric = symmetric
        self.target_tau = target_tau
        self.effective_samplerate = effective_samplerate
        self.domain = domain

        self.log_step = torch.nn.Parameter(init_log_steps(d_state, dt_min, dt_max))
        step = step_scale * torch.exp(self.log_step)

        if structured_initialisation:
            if domain == "frequency":
                N_bins = d_out
            elif domain == "time":
                N_bins = d_in
            else:
                raise NotImplementedError(f"domain {domain} not implemented")
            lambda_unscaled = make_structured_eigenvalues(
                d_state, N_bins, positive_frequencies=init_with_positive_frequencies, 
                domain=domain, target_tau=self.target_tau, effective_samplerate=self.effective_samplerate
            )
            lambda_scaled = torch.cat(((lambda_unscaled[:, 0])[:, None], (lambda_unscaled[:, 1] / step)[:, None]), dim=1)
            Lambda = lambda_scaled
    
        elif log_distributed_frequencies:
            Lambda = make_spectrograms_eigenvalues(d_state, log_distributed_frequencies=log_distributed_frequencies)
            Lambda = Lambda / torch.exp(self.log_step)[:, None]
        else:
            Lambda = make_linear_eigenvalues(d_state, symmetric=self.symmetric, positive_frequencies=init_with_positive_frequencies)

        print("Re_lambda_min / max", (Lambda[:, 0] * step).min().item(), (Lambda[:, 0] * step).max().item())

        self.discretize = discretize_zoh
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.step_scale = step_scale
        self.ensure_stability = ensure_stability
        self.chunk_duration = chunk_duration
        self.d_out = d_out
        self.subsampling_factor = subsampling_factor
        self.samplerate = samplerate
        self.eps_stability = eps_stability
        self.sigmoid_scale = sigmoid_scale

        # --- Sigmoid interval reparametrisation ---
        if ensure_stability == 'sigmoid_interval':
            assert re_lower is not None and re_upper is not None, \
                "re_lower and re_upper must be provided when ensure_stability='sigmoid_interval'"
            assert re_lower < re_upper < 0, \
                f"Need re_lower < re_upper < 0, got re_lower={re_lower}, re_upper={re_upper}"
            self.register_buffer('re_lower', torch.tensor(re_lower, dtype=torch.float32))
            self.register_buffer('re_upper', torch.tensor(re_upper, dtype=torch.float32))

            initial_product = (Lambda[:, 0] * step).clamp(min=re_lower, max=re_upper)
            t = (initial_product - re_lower) / (re_upper - re_lower)
            t = t.clamp(1e-6, 1 - 1e-6)
            Lambda_raw_real = torch.log(t / (1 - t)) / self.sigmoid_scale
            Lambda[:, 0] = Lambda_raw_real

        self.Lambda = torch.nn.Parameter(Lambda)

        assert chunk_duration > subsampling_factor, f"Chunk duration ({chunk_duration}) must be greater than the downsampling factor ({subsampling_factor})"

        if self.input_bias:
            if bias_init == 'zero':
                self.B_bias = torch.nn.Parameter(torch.zeros(d_state, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.B_bias = torch.nn.Parameter(torch.rand(d_state, 2, dtype=torch.float))
        else:
            self.B_bias = torch.nn.Parameter(torch.zeros(d_state, 2, dtype=torch.float), requires_grad=False)

        if self.output_bias:
            if bias_init == 'zero':
                self.C_bias = torch.nn.Parameter(torch.zeros(d_out, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.C_bias = torch.nn.Parameter(torch.rand(d_out, 2, dtype=torch.float))
        else:
            self.C_bias = torch.nn.Parameter(torch.zeros(d_out, 2, dtype=torch.float), requires_grad=False)

        gain = np.sqrt(4 / 12)

        if B_C_init == 'S5':
            lamb, B, C = S5_init(d_in, d_out, d_state)
            self.Lambda.data = lamb
            self.B = torch.nn.Parameter(B, requires_grad=True)
            self.C = torch.nn.Parameter(2 * C, requires_grad=True)
            self.B_bias = torch.nn.Parameter(self.B_bias.data[:self.Lambda.shape[0], ...], requires_grad=True)
            self.log_step = torch.nn.Parameter(init_log_steps(self.Lambda.shape[0], dt_min, dt_max))

            print('S5 init')
            print('A', self.Lambda.shape)
            print('B', self.B.shape)
            print('C', self.C.shape)

        elif B_C_init == 'ones':
            B_r = torch.ones(d_state, d_in)
            B_i = torch.zeros(d_state, d_in)
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))

            if C_C_init == 'convolution':
                print("C_init_with_convolution")
                C_r = torch.eye(d_out, d_state)
                for i in range(d_out):
                    for j in range(d_state):
                        if j == i + 1 or j == i - 1:
                            C_r[i, j] = 1
                C_i = C_r.clone()
            elif C_C_init == 'diagonal':
                C_r = torch.eye(d_out, d_state)
                C_i = torch.zeros(d_out, d_state)
            elif C_C_init == "istft" and d_out == d_in:
                n_grid = np.arange(d_out)[:, None]
                m_grid = np.arange(d_state)[None, :]
                C_init = (1.0 / d_out) * np.exp(1j * 2 * np.pi * n_grid * (m_grid % d_out) / d_out)
                C_r = torch.tensor(C_init.real, dtype=torch.float)
                C_i = torch.tensor(C_init.imag, dtype=torch.float)
            else:
                C_r = torch.empty(d_out, d_state)
                C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
                C_i = torch.empty(d_out, d_state)
                C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T

            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'orthogonal' or B_C_init is None:
            B_r = torch.empty(d_state, d_in)
            B_i = torch.empty(d_state, d_in)
            if d_in == 1:
                B_r = torch.nn.init.normal_(B_r, std=gain)
                B_i = torch.nn.init.normal_(B_i, std=gain)
            else:
                B_r = torch.nn.init.orthogonal_(B_r.T, gain=gain).T
                B_i = torch.nn.init.orthogonal_(B_i.T, gain=gain).T

            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))

            if C_C_init == "istft" and d_out == d_in:
                n_grid = np.arange(d_out)[:, None]
                m_grid = np.arange(d_state)[None, :]
                C_init = (1.0 / d_out) * np.exp(1j * 2 * np.pi * n_grid * (m_grid % d_out) / d_out)
                C_r = torch.tensor(C_init.real, dtype=torch.float)
                C_i = torch.tensor(C_init.imag, dtype=torch.float)
            elif C_C_init == 'convolution':
                C_r = torch.eye(d_out, d_state)
                for i in range(d_out):
                    for j in range(d_state):
                        if j == i + 1 or j == i - 1:
                            C_r[i, j] = 1
                C_i = C_r.clone()
            elif C_C_init == 'diagonal':
                C_r = torch.eye(d_out, d_state)
                C_i = torch.zeros(d_out, d_state)
            else:
                C_r = torch.empty(d_out, d_state)
                C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
                C_i = torch.empty(d_out, d_state)
                C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_uniform_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_uniform_(B_i, nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_uniform_(C_r, nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_uniform_(C_i, nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_normal_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_normal_(B_i, nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_normal_(C_r, nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_normal_(C_i, nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_uniform_(B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_uniform_(B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_uniform_(C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_uniform_(C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_normal_(B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_normal_(B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_normal_(C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_normal_(C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        # print('A', self.Lambda.shape)
        # print('B', self.B.shape)
        # print('C', self.C.shape)
        # print('B_bias', self.B_bias.shape)
        # print('C_bias', self.C_bias.shape)


    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C.shape[-2]))

    def forward_rnn(self, signal, prev_state):
        Lambda_c = as_complex(self.Lambda)

        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        cinput_sequence = signal.type(C_c.dtype)

        step = self.step_scale * torch.exp(self.log_step)
        # print('step', step)
        Lambda_bar, B_bars = self.discretize(
            Lambda_c, B_c, B_bias_c, step, self.input_bias)

        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bar[:, 0])

        # print('Lambda_bar', Lambda_bar.shape)
        # print('input', cinput_sequence)
        # print('B_bar_forward_rnn', B_bar)
        # print('B_bias_bar_forward_rnn', B_bias_bar)
        # print('C_c', C_c)

        Bu = B_bar @ cinput_sequence + B_bias_bar
        x = Lambda_bar * prev_state + Bu
        y = C_c @ x + C_bias_c

        if self.complex_output:
            y_out = y
        else:
            y_out = y.real
        return y_out, x

    def forward(self, signal):

        step = self.step_scale * torch.exp(self.log_step)

        if self.ensure_stability == 'sigmoid_interval':
            # --- Differentiable reparametrisation with internal scaling ---
            # We use an internal scaling factor of S = self.sigmoid_scale to translate AdamW's updates
            # into physically significant but stable updates.
            width = self.re_upper - self.re_lower
            product = self.re_lower + width * torch.sigmoid(self.sigmoid_scale * self.Lambda[:, 0])
            # product = Re(Lambda_c) * step, guaranteed in [re_lower, re_upper]
            effective_real = product / step
            # Build complex Lambda_c with constrained real part and original imaginary part
            Lambda_c = torch.complex(effective_real, self.Lambda[:, 1])
        else:
            eps_stability = self.eps_stability / torch.min(step)

            with torch.no_grad():
                if self.ensure_stability == 'relu':
                    self.Lambda.data[:, 0] = -F.relu(- (self.Lambda.data[:, 0] + eps_stability)) - eps_stability  # stability : lambda real < - epsilon --> - (lambda + epsilon) > 0
                    # self.Lambda.data[:, 0] = -F.relu(-self.Lambda.data[:, 0])
                    # Lambda_c.real = -F.relu(-Lambda_c.real) # Ensure stability
                elif self.ensure_stability == 'abs':
                    self.Lambda.data[:, 0] = -torch.abs(self.Lambda.data[:, 0]).clamp(min=eps_stability)
                    # self.Lambda.data[:, 0] = -torch.abs(self.Lambda.data[:, 0])
                    # Lambda = torch.complex(-torch.abs(Lambda.real), Lambda.imag)

            Lambda_c = as_complex(self.Lambda)

       


        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        Lambda_bars, B_bars = self.discretize(
            Lambda_c, B_c, B_bias_c, step, self.input_bias)
        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bars[:, 0])
        # forward = apply_ssm



        B,T,num_channels = signal.shape

        c = self.chunk_duration
        h = self.subsampling_factor



        total_num_samples = math.ceil(T/h)

        out_dtype = torch.complex64 if self.complex_output else torch.float32
        output = torch.zeros(B, total_num_samples, self.d_out, device=signal.device, dtype=out_dtype)

        current_index = 0
        chunks = torch.split(signal, c, dim=1)
        last_state = None
        offset = 0

        for chunk in chunks:
            chunk_len = chunk.shape[1]
            if self.training:
                last_state, out = checkpoint(
                    apply_ssm_progressive,
                    Lambda_bars, B_bar, B_bias_bar, C_c, C_bias_c, 
                    chunk, 
                    self.complex_output, 
                    last_state, 
                    h,
                    offset,
                    use_reentrant=False,
                )
            else:
                # Mode évaluation/validation (pas de checkpointing)
                last_state, out = apply_ssm_progressive(
                    Lambda_bars, B_bar, B_bias_bar, C_c, C_bias_c, 
                    chunk, 
                    self.complex_output, 
                    last_state=last_state, 
                    subsampling_factor=h,
                    offset=offset
                )

            num_samples = out.shape[1]
            output[:, current_index : current_index + num_samples, :] = out
            current_index += num_samples

            offset = (h - (chunk_len - offset) % h) % h

        return output




class Progressive_SSM_FT(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 dt_min: float,
                 dt_max: float,
                 step_scale: float = 1.0,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=False,
                 og = False,
                 B_C_init= None,
                 C_C_init= None,
                 ensure_stability='abs',
                 symmetric=False,
                 chunk_duration = 264600,
                 subsampling_factor = 1,
                 log_distributed_frequencies= False,
                 samplerate = 44100.0,
                 eps_stability: float = 1e-3,
                 re_lower: float = None,
                 re_upper: float = None,
                 sigmoid_scale: float = 1.0,
                 structured_initialisation = False,
                 init_with_positive_frequencies = True,
                 domain = "frequency",
                 phase_correction = False,
                 synthesis_window = None,
                 n_hop_for_ola = None,
                 target_tau = None,
                 effective_samplerate = None
                 ): 
        """The Fast Fourier Transform Convolution SSM (Progressive_SSM_FT)"""
        super().__init__()
        self.symmetric = symmetric
        self.target_tau = target_tau
        self.effective_samplerate = effective_samplerate
        self.domain = domain

        self.log_step = torch.nn.Parameter(init_log_steps(d_state, dt_min, dt_max))
        step = step_scale * torch.exp(self.log_step)

        if structured_initialisation:
            if domain == "frequency":
                N_bins = d_out
            elif domain == "time":
                N_bins = d_in
            else:
                raise NotImplementedError(f"domain {domain} not implemented")
            lambda_unscaled = make_structured_eigenvalues(
                d_state, N_bins, positive_frequencies=init_with_positive_frequencies, 
                domain=domain, target_tau=self.target_tau, effective_samplerate=self.effective_samplerate
            )
            lambda_scaled = torch.cat(((lambda_unscaled[:, 0])[:, None], (lambda_unscaled[:, 1] / step)[:, None]), dim=1)
            Lambda = lambda_scaled
        elif og:
            Lambda = make_linear_eigenvalues(d_state, symmetric=self.symmetric, positive_frequencies=init_with_positive_frequencies)
        elif log_distributed_frequencies:
            Lambda = make_spectrograms_eigenvalues(d_state, log_distributed_frequencies=log_distributed_frequencies)
            Lambda = Lambda / torch.exp(self.log_step)[:, None]
        else:
            Lambda = make_linear_eigenvalues(d_state, symmetric=self.symmetric, positive_frequencies=init_with_positive_frequencies)

        self.discretize = discretize_zoh
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.step_scale = step_scale
        self.ensure_stability = ensure_stability
        self.chunk_duration = chunk_duration
        self.d_out = d_out
        self.subsampling_factor = subsampling_factor
        self.samplerate = samplerate
        self.eps_stability = eps_stability
        self.sigmoid_scale = sigmoid_scale

        # --- Sigmoid interval reparametrisation ---
        if ensure_stability == 'sigmoid_interval':
            assert re_lower is not None and re_upper is not None, \
                "re_lower and re_upper must be provided when ensure_stability='sigmoid_interval'"
            assert re_lower < re_upper < 0, \
                f"Need re_lower < re_upper < 0, got re_lower={re_lower}, re_upper={re_upper}"
            self.register_buffer('re_lower', torch.tensor(re_lower, dtype=torch.float32))
            self.register_buffer('re_upper', torch.tensor(re_upper, dtype=torch.float32))

            initial_product = (Lambda[:, 0] * step).clamp(min=re_lower, max=re_upper)
            t = (initial_product - re_lower) / (re_upper - re_lower)
            t = t.clamp(1e-6, 1 - 1e-6)
            Lambda_raw_real = torch.log(t / (1 - t)) / self.sigmoid_scale
            Lambda[:, 0] = Lambda_raw_real

        self.Lambda = torch.nn.Parameter(Lambda)

        if self.input_bias:
            if bias_init == 'zero':
                self.B_bias = torch.nn.Parameter(torch.zeros(d_state, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.B_bias = torch.nn.Parameter(torch.rand(d_state, 2, dtype=torch.float))
        else:
            self.B_bias = torch.nn.Parameter(torch.zeros(d_state, 2, dtype=torch.float), requires_grad=False)

        if self.output_bias:
            if bias_init == 'zero':
                self.C_bias = torch.nn.Parameter(torch.zeros(d_out, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.C_bias = torch.nn.Parameter(torch.rand(d_out, 2, dtype=torch.float))
        else:
            self.C_bias = torch.nn.Parameter(torch.zeros(d_out, 2, dtype=torch.float), requires_grad=False)

        gain = np.sqrt(4 / 12)

        if B_C_init == 'S5':
            lamb, B, C = S5_init(d_in, d_out, d_state)
            self.Lambda.data = lamb
            self.B = torch.nn.Parameter(B, requires_grad=True)
            self.C = torch.nn.Parameter(2 * C, requires_grad=True)
            self.B_bias = torch.nn.Parameter(self.B_bias.data[:self.Lambda.shape[0], ...], requires_grad=True)
            self.log_step = torch.nn.Parameter(init_log_steps(self.Lambda.shape[0], dt_min, dt_max))

        elif B_C_init == 'ones':
            B_r = torch.ones(d_state, d_in)
            B_i = torch.zeros(d_state, d_in)
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))

            if C_C_init == 'convolution':
                C_r = torch.eye(d_out, d_state)
                for i in range(d_out):
                    for j in range(d_state):
                        if j == i + 1 or j == i - 1:
                            C_r[i, j] = 1
                C_i = C_r.clone()
            elif C_C_init == 'diagonal':
                C_r = torch.eye(d_out, d_state)
                C_i = torch.zeros(d_out, d_state)
            elif C_C_init == "istft" and d_out == d_in:
                n_grid = np.arange(d_out)[:, None]
                m_grid = np.arange(d_state)[None, :]
                C_init = (1.0 / d_out) * np.exp(1j * 2 * np.pi * n_grid * (m_grid % d_out) / d_out)
                C_r = torch.tensor(C_init.real, dtype=torch.float)
                C_i = torch.tensor(C_init.imag, dtype=torch.float)
            else:
                C_r = torch.empty(d_out, d_state)
                C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
                C_i = torch.empty(d_out, d_state)
                C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T

            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'orthogonal' or B_C_init is None:
            B_r = torch.empty(d_state, d_in)
            B_i = torch.empty(d_state, d_in)
            if d_in == 1:
                B_r = torch.nn.init.normal_(B_r, std=gain)
                B_i = torch.nn.init.normal_(B_i, std=gain)
            else:
                B_r = torch.nn.init.orthogonal_(B_r.T, gain=gain).T
                B_i = torch.nn.init.orthogonal_(B_i.T, gain=gain).T

            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))

            if C_C_init == "istft" and d_out == d_in:
                n_grid = np.arange(d_out)[:, None]
                m_grid = np.arange(d_state)[None, :]
                C_init = (1.0 / d_out) * np.exp(1j * 2 * np.pi * n_grid * (m_grid % d_out) / d_out)
                C_r = torch.tensor(C_init.real, dtype=torch.float)
                C_i = torch.tensor(C_init.imag, dtype=torch.float)
            elif C_C_init == 'convolution':
                C_r = torch.eye(d_out, d_state)
                for i in range(d_out):
                    for j in range(d_state):
                        if j == i + 1 or j == i - 1:
                            C_r[i, j] = 1
                C_i = C_r.clone()
            elif C_C_init == 'diagonal':
                C_r = torch.eye(d_out, d_state)
                C_i = torch.zeros(d_out, d_state)
            else:
                C_r = torch.empty(d_out, d_state)
                C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
                C_i = torch.empty(d_out, d_state)
                C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_uniform_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_uniform_(B_i, nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_uniform_(C_r, nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_uniform_(C_i, nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_normal_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_normal_(B_i, nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_normal_(C_r, nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_normal_(C_i, nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_uniform_(B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_uniform_(B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_uniform_(C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_uniform_(C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_normal_(B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_normal_(B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_normal_(C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_normal_(C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

    def forward(self, signal):
        step = self.step_scale * torch.exp(self.log_step)

        if self.ensure_stability == 'sigmoid_interval':
            width = self.re_upper - self.re_lower
            product = self.re_lower + width * torch.sigmoid(self.sigmoid_scale * self.Lambda[:, 0])
            effective_real = product / step
            Lambda_c = torch.complex(effective_real, self.Lambda[:, 1])
        else:
            eps_stability = self.eps_stability / torch.min(step)
            with torch.no_grad():
                if self.ensure_stability == 'relu':
                    self.Lambda.data[:, 0] = -F.relu(- (self.Lambda.data[:, 0] + eps_stability)) - eps_stability
                elif self.ensure_stability == 'abs':
                    self.Lambda.data[:, 0] = -torch.abs(self.Lambda.data[:, 0]).clamp(min=eps_stability)
            Lambda_c = as_complex(self.Lambda)

        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        Lambda_bars, B_bars = self.discretize(Lambda_c, B_c, B_bias_c, step, self.input_bias)
        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bars[:, 0])

        return apply_ssm_ft(
            Lambda_bars, B_bar, B_bias_bar, C_c, C_bias_c, 
            signal, self.complex_output, self.subsampling_factor
        )