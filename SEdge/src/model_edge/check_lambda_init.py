from init import make_spectrograms_eigenvalues
import torch
from ssm import as_complex
from matplotlib import pyplot as plt
import numpy as np

d_state = 10
log_distributed_frequencies= True

log_scale = torch.linspace(0, np.log(1+np.pi), d_state)
omega = (torch.exp(log_scale) - 1).to(torch.float32)

real_comparison = -torch.ones(d_state)/2 * torch.cos(omega)
imag_comparison = -torch.ones(d_state)/2 * torch.sin(omega)

comparison = torch.stack((real_comparison, imag_comparison), dim=-1)
comparison_c = as_complex(comparison)

Lambda = make_spectrograms_eigenvalues(d_state, mel=mel)
Lambda_c = as_complex(Lambda)

plt.scatter(Lambda_c.real, Lambda_c.imag, label="init", marker="x")
plt.scatter(comparison_c.real, comparison_c.imag, label = "ground truth", alpha = 0.3)
plt.legend()
plt.savefig("./eigenvalues_comparison.png")