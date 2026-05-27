import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,10)/10
y = np.array([
    14490, 14672, 14952, 15192, 15356, 15656, 15714, 15258, 15664
    ])

plt.plot(x,y)
plt.savefig(f'/home/adubois/openunmix/OpenUnmix/fig/test_memory.png')
