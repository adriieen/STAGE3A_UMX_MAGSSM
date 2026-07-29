import numpy as np
beta = 0
im_lambda = np.array([np.pi/4, np.pi/2, 3*np.pi])
w = np.zeros(im_lambda.shape)
high_freq = im_lambda > np.pi
w[high_freq] = im_lambda[high_freq] - np.pi + beta
print(high_freq)
print(w)