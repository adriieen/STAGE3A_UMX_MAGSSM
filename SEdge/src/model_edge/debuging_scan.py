from ssm_bis import SSM, Progressive_SSM
import torch
import numpy as np
import matplotlib.pyplot as plt




signal_length = 100000
chunk_duration = 1254
subsampling_factor = 100



example_spectrogram = torch.randn(signal_length).to(torch.float32)

example_spectrogram = example_spectrogram[None, :, None] # 1, T, 1 



# The different scans


# lambda1 = -1/2 * np.exp(1j*np.pi/4)
# lambda2 = -1/2 * np.exp(1j*np.pi/2)

# Lambda = np.array([lambda1, lambda2])

# lambda_real = np.expand_dims(Lambda.real, axis=1)
# lambda_imag = np.expand_dims(Lambda.imag, axis=1)
# Lambda = np.concatenate((lambda_real, lambda_imag), axis=1)
# Lambda = torch.tensor(Lambda, dtype=torch.float)


ssm_standard = Progressive_SSM(
    d_in = 1,
    d_state = 2,
    d_out = 1,
    dt_min = 1e-3,
    dt_max = 1e-1,
    chunk_duration = signal_length,
    subsampling_factor = 1,
    B_C_init = "ones",
    C_C_init='convolution',
)

ssm_progressive = Progressive_SSM(
    d_in = 1,
    d_state = 2,
    d_out = 1,
    dt_min = 1e-3,
    dt_max = 1e-1,
    B_C_init = "ones",
    C_C_init = "convolution",
    chunk_duration = chunk_duration,
    subsampling_factor = subsampling_factor

)

out_standard, chunks_standard = ssm_standard(example_spectrogram)

# out_progressive = ssm_progressive(example_spectrogram)[:,::subsampling_factor,:]
out_progressive, chunks_progressive = ssm_progressive(example_spectrogram)


print(f"Standard SSM output shape: {out_standard.shape}")
print(f"Progressive SSM output shape: {out_progressive.shape}")


out_standard_downsampled = out_standard[:, ::subsampling_factor, :]

print(f"Standard SSM output downsampled shape: {out_standard_downsampled.shape}")



print(f"Chunk shapes -- standard / progressive : {len(chunks_standard)} chunks, {len(chunks_progressive)} chunks")

# compare chunks



# compare 
wrong = torch.where(out_standard_downsampled != out_progressive, True, False)

print(f"The two outputs are different: {wrong.sum()}")


ranks = wrong.nonzero(as_tuple=True)

#save to file
#get ranks of wrong elements

with open("wrong_ranks.txt", "w") as f:
    for rank in ranks[1]:
        f.write(f"{rank}\n")    



# find the matching that is happening

# with open("indices_match.txt", "w") as f:
#     for i, sample in enumerate(out_progressive[0,:,0]):
#         diff =  torch.mean(torch.abs(out_standard[0,:,0] - sample))Ò
#         #index = torch.argmin(diff)
        
#         f.write(f"Sampled index {i} -> Original index {index}; difference = {diff}\n")
        

plt.plot(out_progressive[0,-100:-1,0].detach().numpy(), label = "processed by chunks")
plt.plot(out_standard_downsampled[0,-100:-1,0].detach().numpy(), label = "processed entirely", linestyle = "--")
plt.legend()
plt.savefig("./comparison.png")

