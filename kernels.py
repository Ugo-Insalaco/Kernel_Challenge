import numpy as np

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N, M = (X.shape[0], Y.shape[0])
        X = X.reshape(N, -1)
        Y = Y.reshape(M, -1)
        K = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                K[i, j] = np.exp(-np.linalg.norm(X[i] - Y[j])**2/(2*self.sigma**2))
        # K = np.exp(-np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)**2/(2*self.sigma**2)) # N x M
        return K
    
class Linear:
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        K = X @ Y.T
        return K ## Matrix of shape NxM
    
