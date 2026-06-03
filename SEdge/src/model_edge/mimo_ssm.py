import torch

from typing import Optional

from .ssm_bis import SSM, Progressive_SSM

class MIMOSSM(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 step_scale: float = 1.0,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=True,
                 B_C_init='orthogonal',
                 C_C_init= None, 
                 stability='abs',
                 progressive = False,
                 chunk_duration : Optional[int] = None,
                 subsampling_factor = 1,
                 log_distributed_frequencies = False

                ):
        
        super().__init__()
        self.d_in = d_in
        self.d_state = d_state
        self.d_out = d_out
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.step_scale = step_scale
        self.previous_step_scale = step_scale
        self.complex_output = complex_output

        if not progressive:

            self.seq = SSM(
                d_in,
                d_state,
                d_out,
                dt_min,
                dt_max,
                step_scale,
                input_bias=input_bias,
                bias_init=bias_init,
                output_bias=output_bias,
                complex_output=complex_output,
                B_C_init=B_C_init,
                ensure_stability=stability,
                subsampling_factor = subsampling_factor
            )

        else:
            self.seq = Progressive_SSM(
                d_in,
                d_state,
                d_out,
                dt_min,
                dt_max,
                step_scale,
                input_bias=input_bias,
                bias_init=bias_init,
                output_bias=output_bias,
                complex_output=complex_output,
                B_C_init=B_C_init,
                C_C_init = C_C_init,
                ensure_stability=stability,
                chunk_duration = chunk_duration,
                subsampling_factor = subsampling_factor,
                log_distributed_frequencies = log_distributed_frequencies

            )
            

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self, signal):
        # return torch.vmap(lambda s: self.seq(s))(signal)
        return self.seq(signal)

    def set_step_scale(self, step_scale):
        self.step_scale = step_scale
        self.seq.step_scale = step_scale