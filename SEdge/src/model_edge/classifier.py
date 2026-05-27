import torch
import numpy as np

from .sequence_layer import SequenceLayer
from .random_zero_order_hold import RandomZeroOrderHold

class SC_Model_classifier(torch.nn.Module):
    def __init__(self, *, input_size=1, classes=35, hidden_sizes=[], output_sizes=[], ZeroOrderHoldRegularization=[],
                 input_bias=False, bias_init='zero',
                 output_bias=False, complex_output=False,
                 norm=False, norm_type='bn', B_C_init='orthogonal', stability='relu', trainable_SkipLayer=False, act='RELu',dropout=0.0, **kwargs):
        super(SC_Model_classifier, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.trainable_skip_layer = trainable_SkipLayer
        self.act=act
        self.dropout = dropout

        if 'n_layer' in kwargs:
            raise ValueError(
                'n_layer is deprecated in SC_Model_classifier, use lists of hidden_sizes and output_sizes instead')

        if len(kwargs) > 0:
            raise ValueError('Unknown keyword arguments: ' + str(kwargs))

        if not isinstance(hidden_sizes, list):
            raise ValueError('hidden_sizes must be a list')

        if not isinstance(output_sizes, list):
            raise ValueError('output_sizes must be a list')

        if not isinstance(ZeroOrderHoldRegularization, list):
            if isinstance(ZeroOrderHoldRegularization, float):
                ZeroOrderHoldRegularization = [ZeroOrderHoldRegularization]*len(hidden_sizes)
            elif ZeroOrderHoldRegularization is None:
                ZeroOrderHoldRegularization = []
            else:
                raise ValueError(
                    'ZeroOrderHoldRegularization must be a list of floats, or a float or None or empty list')

        if len(hidden_sizes) != len(output_sizes):
            raise ValueError(
                'hidden_sizes and output_sizes must have the same length')

        if len(ZeroOrderHoldRegularization) != 0 and len(ZeroOrderHoldRegularization) != len(hidden_sizes):
            print(ZeroOrderHoldRegularization)
            print(hidden_sizes)
            raise ValueError('ZeroOrderHoldRegularization must have the same length as hidden_sizes')

        sequence_layers = []
        sequence_layers.append(SequenceLayer(d_in=input_size,
                                             d_state=hidden_sizes[0],
                                             d_out=output_sizes[0],
                                             input_bias=self.input_bias,
                                             bias_init=bias_init,
                                             output_bias=self.output_bias, norm=norm, norm_type=norm_type,
                                             complex_input=False,
                                             complex_output=self.complex_output,
                                             B_C_init=B_C_init, stability=stability,
                                             trainable_SkipLayer=self.trainable_skip_layer,
                                             act=self.act,
                                             dropout=dropout,
                                             ))
        for i in range(1, len(hidden_sizes)):
            sequence_layers.append(SequenceLayer(d_in=output_sizes[i-1],
                                                 d_state=hidden_sizes[i],
                                                 d_out=output_sizes[i],
                                                 input_bias=self.input_bias,
                                                 bias_init=bias_init,
                                                 output_bias=self.output_bias, norm=norm, norm_type=norm_type,
                                                 complex_input=False,
                                                 complex_output=self.complex_output,
                                                 B_C_init=B_C_init, stability=stability,
                                                 trainable_SkipLayer=self.trainable_skip_layer,
                                                 act=self.act,
                                                 dropout=dropout,
                                                 ))

        Reg_layers = []
        for i,_ in enumerate(ZeroOrderHoldRegularization):
            Reg_layers.append(RandomZeroOrderHold(
                ZeroOrderHoldRegularization[i]))

        self.seq = torch.nn.Sequential(*sequence_layers)

        if len(ZeroOrderHoldRegularization) > 0:
            self.reg = torch.nn.Sequential(*Reg_layers)
        else:
            self.reg = torch.nn.Sequential(*([torch.nn.Identity()]*len(self.seq)))

        self.decoder = torch.nn.Linear(output_sizes[-1], self.classes)
        # print('decoder norm', torch.norm(self.decoder.weight.data, dim=1, keepdim=True))
        # print(np.sqrt(4/12))
        self.decoder.weight.data = np.sqrt(4/12)*self.decoder.weight.data/torch.norm(self.decoder.weight.data, dim=1, keepdim=True)

        self.input_norm = torch.nn.BatchNorm1d(input_size, affine=True, momentum=1e-2)  # works

    def forward(self, inputs):
        # inputs = self.input_norm(inputs.transpose(1, 2)).transpose(1, 2)

        for i,_ in enumerate(self.seq):
            inputs = self.seq[i](inputs)
            inputs = self.reg[i](inputs)

        output_mean = torch.mean(inputs, dim=1)
        
        logits = self.decoder(output_mean)
        return logits

    def set_step_scale(self, step_scales, previous_step_scales=None):
        # self.step_scales = step_scales
        if previous_step_scales is None:
            previous_step_scales = step_scales
        for previous_step_scale, step_scale, layer in zip(previous_step_scales, step_scales, self.seq):
            layer.set_step_scale(step_scale, previous_step_scale)


    def get_number_of_parameters(self):
        params = 0
        for layer in self.seq:
            params += layer.get_number_of_parameters()

        params += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        params += 2 # Input BN

        return params

    def get_number_of_MACs(self):
        n_MACs = 0
        for layer in self.seq:
            n_MACs += layer.get_number_of_MACs()

        n_MACs += self.decoder.in_features/self.seq[-1].s5.step_scale       ## reduction layer, only additions
        n_MACs += self.decoder.in_features * self.decoder.out_features/16000 ## decoder layer (16000 samples)

        n_MACs += 1 # Input BN

        return n_MACs
