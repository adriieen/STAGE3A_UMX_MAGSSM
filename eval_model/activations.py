import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
models = {

    "256 states - hidden size =128" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/true_magssm/256states/hid128_fixed_init/vocals.pth",
    "256 states - hidden size = 512" : "/home/adubois/openunmix/OpenUnmix/outputs/500ep/true_magssm/256states/hid512_fixed_init/vocals.pth"
}

sampling_rate = 44100


for model in models:
    state_dict = torch.load(models[model], map_location='cpu')
    
    log_sigma = state_dict["magssm.mimo.seq.log_sigma"]
    log_omega = state_dict["magssm.mimo.seq.log_omega"]
    B = state_dict["magssm.mimo.seq.B"]
    C = state_dict["magssm.mimo.seq.C"]


    Lambda = torch.complex(-torch.exp(log_sigma), torch.exp(log_omega))


    freqs = torch.linspace(0, sampling_rate/2, 10000)

    unit_circle = torch.exp(2*1j*torch.pi*freqs)

    
    # calculate transfer function

    d_out = C.shape[0]
    for idx in range(d_out):

        C_idx = C[idx]
        C_complex = torch.complex(C_idx[:, 0], C_idx[:, 1])


        print(C_complex.shape, B.shape)
      


        B_complex = torch.complex(B[...,0], B[...,1])

        A = Lambda[:, None] * torch.eye(Lambda.shape[0])

        A = A[:,:,None]
        A = A.tile((1,1,freqs.shape[0]))

        print(B_complex.shape, Lambda.shape, A.shape)

        matrix = unit_circle[None, None,:] - A

        print(matrix.shape)
        
        inv = torch.linalg.inv(matrix) # N_states x N_states x N_freqs

        print(inv.shape)

        right_product = inv.permute(2,0,1) @ B_complex # N_freqs x Nb_states 

        print(right_product.shape)
        
        transfer = C_complex @ right_product.permute(1,0) # N_freqs x 1
        
        





    plt.figure()
    plt.plot(freqs, Transfers)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"Activations : {model}")
    plt.savefig(f'/home/adubois/openunmix/OpenUnmix/fig/activations_{model}.png')
    plt.close()
    
