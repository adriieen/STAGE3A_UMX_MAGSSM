import numpy as np
L=np.arange(10)
print(L[0:-1])

import torch    
y = torch.randn(30)
y = y.reshape((2,3) + (5,))
print(y)
print(y.shape)