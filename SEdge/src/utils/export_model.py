import torch

import numpy as np


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


def export_layer_parameters(model_trained, step_scale_list=[1]*10):
    layer_dict = {}
    layer_dict_c = {}
    model_trained = model_trained.cpu()

    for i, mimo_ssm in enumerate(model_trained.seq):
        A_continious = mimo_ssm.s5.seq.Lambda.detach()
        A_continious[:, 0] = -abs(A_continious[:, 0])
        # zero_real_idx = A_continious[:,0] == 0
        # A_continious[zero_real_idx,0] = 1e-4

        B_continious = mimo_ssm.s5.seq.B.detach()
        B_bias_continious = mimo_ssm.s5.seq.B_bias.detach()
        if mimo_ssm.norm:
            # print('Batch norm active')
            runnig_var = mimo_ssm.attn_norm.running_var.detach()
            runnig_mean = mimo_ssm.attn_norm.running_mean.detach()
            scale = mimo_ssm.attn_norm.weight.detach()
            bias = mimo_ssm.attn_norm.bias.detach()
            scale = scale / torch.sqrt(runnig_var + 1e-5)
            # print('scale:', scale)
            bias = bias - runnig_mean * scale
            B_bias_continious = B_bias_continious + torch.sum(B_continious*bias.view(1, -1, 1), dim=1)
            B_continious = B_continious * scale.view(1, -1, 1)

        C = mimo_ssm.s5.seq.C.detach()
        C_bias = mimo_ssm.s5.seq.C_bias.detach()  # .numpy()

        SkipLayer = getattr(mimo_ssm, 'skipLayer', None)
        if SkipLayer is not None:
            SkipLayer = SkipLayer.weight.detach().numpy()
            # print('real skiplayer:', SkipLayer.shape)
            SkipLayer = SkipLayer + 1j * np.zeros_like(SkipLayer)  # complex representation with zero imaginary part
        else:
            SkipLayer = np.ones((1))
            SkipLayer = SkipLayer + 1j * np.zeros_like(SkipLayer)  # complex representation with zero imaginary part

        # try:
        #     SkipLayer = mimo_ssm.skipLayer.weight.detach().numpy()
        #     # print('real skiplayer:', SkipLayer.shape)
        # except:
        #     SkipLayer = np.ones((C.shape[0], B_continious.shape[1]))
            # print('ones skiplayer:', SkipLayer.shape)
        

        A_continious = as_complex(A_continious)
        B_continious = as_complex(B_continious)
        B_bias_continious = as_complex(B_bias_continious)
        C_continious = as_complex(C).numpy()
        C = as_complex(C)
        C_bias = as_complex(C_bias)

        step_scale = step_scale_list[i]
        delta = torch.exp(mimo_ssm.s5.seq.log_step).detach()
        # print(delta.shape)
        # print(step_scale)
        step = step_scale*delta
        A_d, Bb_d = mimo_ssm.s5.seq.discretize(
            A_continious, B_continious, B_bias_continious, step, mimo_ssm.s5.seq.input_bias)

        B_d = Bb_d[:, 0:-1]
        B_bias_d = Bb_d[:, -1]
        # convert to numpy real and imag parts
        # A_d = torch.stack((A_d.real, A_d.imag), dim=-1).detach().numpy()
        # B_d = torch.stack((B_d.real, B_d.imag), dim=-1).detach().numpy()
        # B_bias_d = torch.stack((B_bias_d.real, B_bias_d.imag), dim=-1).detach().numpy()
        # C = torch.stack((C.real, C.imag), dim=-1).detach().numpy()
        # C_bias = torch.stack((C_bias.real, C_bias.imag), dim=-1).detach().numpy()

        # print(B_continious.shape)
        # print(B_bias_d.shape)
        # print(C_bias.shape)
        # print((delta.unsqueeze(1)*B_continious).numpy().shape)
        act:str = str(mimo_ssm.act)
        act = act.split('(')[0]
        layer_dict_c[i] = {f'A': (delta * A_continious).numpy(),
                           f'B': (delta.unsqueeze(1) * B_continious).numpy(),
                           f'B_bias': (delta.unsqueeze(1) * B_bias_continious).numpy(),
                           f'C': C_continious, f'C_bias': C_bias, 'SkipLayer': SkipLayer,
                           f'Activation': act
                           }
        # layer_dict_c[i] = {f'A': (A_continious).numpy(), f'B': (B_continious).numpy(), f'B_bias': (B_bias_continious).numpy(), f'C': C_continious, f'C_bias': C_bias, 'SkipLayer': SkipLayer}
        # layer_dict_c[i] = {f'A': (A_continious).numpy(), f'B': B_continious.numpy(), f'B_bias': B_bias_continious.numpy(), f'C': C_continious, f'C_bias': C_bias}
        layer_dict[i] = {
            f'A': A_d.detach().numpy(),
            f'B': B_d.detach().numpy(),
            f'B_bias': B_bias_d.detach().numpy(),
            f'C': C.detach().numpy(),
            f'C_bias': C_bias.detach().numpy(),
            'SkipLayer': SkipLayer, 'step_scale': step_scale,
             f'Activation': act}

    # Add final Linear Layer
    layer_dict[len(layer_dict)] = {'W': model_trained.decoder.weight.detach().numpy(),
                                   'b': model_trained.decoder.bias.detach().numpy()}

    return layer_dict, layer_dict_c
