import torch
import torch.nn.functional as F

import numpy as np

from .mimo_ssm import MIMOSSM
from .channel_dropout import ChannelDropout
from .batch_norm_transposed import ComplexBatchNorm1dT, BatchNorm1dT

class SequenceLayer(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 step_scale: float = 1.0,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 norm=False,
                 norm_type='bn',
                 complex_input=False,
                 complex_output=False,
                 B_C_init='orthogonal',
                 stability='abs',
                 trainable_SkipLayer=False,
                 dropout=0.0,
                 act='RELu'
                 ):
        super().__init__()
        self.s5 = MIMOSSM(d_in, d_state, d_out, step_scale=step_scale,
                          input_bias=input_bias, bias_init=bias_init,
                          output_bias=output_bias,
                          complex_output=complex_output, B_C_init=B_C_init, stability=stability)
        self.d_in=d_in
        self.d_state=d_state
        self.d_out=d_out
        self.step_scale=step_scale
        self.input_bias=input_bias
        self.bias_init=bias_init
        self.output_bias=output_bias
        self.norm=norm
        self.norm_type=norm_type
        self.complex_input=complex_input
        self.complex_output=complex_output
        self.B_C_init=B_C_init
        self.stability=stability
        self.trainable_SkipLayer=trainable_SkipLayer
        self.dropout=dropout
        self.act=act

        self.trainable_SkipLayer = trainable_SkipLayer
        if self.trainable_SkipLayer:
            self.skipLayer = torch.nn.Linear(d_in, d_out, bias=False)
            self.skipLayer.weight.data = np.sqrt(4/12)*self.skipLayer.weight.data/torch.norm(self.skipLayer.weight.data, dim=1, keepdim=True)
            # self.skipLayer.weight.data = torch.nn.init.orthogonal_(self.skipLayer.weight.data, gain=np.sqrt(4/12)) 

        
        self.dropout = ChannelDropout(p=self.dropout, inplace=False) if dropout > 0 else torch.nn.Identity()


        if norm and (norm_type == 'ln'):
            if complex_input:
                class LayerNormComplex(torch.nn.Module):
                    # init function
                    def __init__(self, d_in):
                        super().__init__()
                        self.ln_real = torch.nn.LayerNorm(d_in)
                        self.ln_imag = torch.nn.LayerNorm(d_in)

                    def forward(self, x):
                        return self.ln_real.forward(x.real) + 1j*self.ln_imag.forward(x.imag)

                self.attn_norm = LayerNormComplex(d_in)
            else:
                self.attn_norm = torch.nn.LayerNorm(d_in)

        elif norm and (norm_type == 'bn'):
            # batch norm 1d needs NxCxL input shape
            if complex_input:
                self.attn_norm = ComplexBatchNorm1dT(d_in)
            else:
                self.attn_norm = BatchNorm1dT(d_in)

        else:
            self.attn_norm = torch.nn.Identity()

        if act == 'RELu':
            self.act = torch.nn.ReLU()
        elif act == 'LeakyRELu':
            self.act = torch.nn.LeakyReLU()
        elif act == 'Identity':
            self.act = torch.nn.Identity()
        else:
            print(f'act: {act}')
            raise NotImplementedError('Activation function not implemented')

    def forward(self, x):

        step_scale = self.s5.step_scale
        previous_step_scale = self.s5.previous_step_scale

        if step_scale != previous_step_scale:
            x = x[:, ::int(step_scale/previous_step_scale), :]

        if self.trainable_SkipLayer:
            res = self.skipLayer(x.clone())
        else:
            res = x.clone()
            
        fx = self.attn_norm(x)
        out = self.s5(fx)

        if self.s5.complex_output:
            x = (self.act(out.real) + 1j*self.act(out.imag)) + res
        else:
            x = self.dropout(self.act(out.real)) + res
            # x = F.leaky_relu(out).real + res
        # x = x/np.sqrt(2)
        return x

    def set_step_scale(self, step_scale, previous_step_scale=None):
        self.step_scale = step_scale
        self.previous_step_scale = previous_step_scale

        if previous_step_scale is None:
            self.s5.previous_step_scale = step_scale
            self.s5.set_step_scale(step_scale)
        else:
            self.s5.previous_step_scale = previous_step_scale
            self.s5.set_step_scale(step_scale)

    def get_number_of_parameters(self):
        # analytical calculation of the number of parameters
        # B/ in Bias Matrix element C
        B = self.d_in*self.d_state*2
        B_bias = self.d_state*2

        # A matrix element C
        A = self.d_state*2

        # C/ out Bias matrix element C
        C = self.d_state*self.d_out*2
        C_bias = self.d_out*2

        # Skip Layer element R
        skip = 0
        if self.trainable_SkipLayer:
            skip = self.d_in*self.d_out
        
        return A + B + B_bias + C + C_bias + skip





    # function the get the number of MUltipliy-Add operations of the model
    def get_number_of_MACs(self):
        # analytical calculation of the number of parameters
        # B Matrix element C @ u element R , bias can be ignored
        B = self.d_in*self.d_state*2
        # A matrix element C @ x element C
        A = self.d_state*4
        # C Matrix element C @ x element C, bias can be ignored, but output is elemet R
        C = self.d_state*self.d_out*2

        # Skip Layer element R
        skip = self.d_out
        if self.trainable_SkipLayer:
            skip = self.d_in*self.d_out

        # now combine based on step scale
        update_macs = A+B
        output_macs = C+skip

        return (update_macs+output_macs)/self.s5.step_scale
