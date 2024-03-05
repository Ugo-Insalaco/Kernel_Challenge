import numpy as np
import tqdm
import os
import fast_histogram
class CacheKernel:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cached_K = None

    def __call__(self, use_cache=True):
        if use_cache:
            if self.cached_K is not None:
                return self.cached_K
            cache_exists = os.path.isfile(self.cache_file)
            if cache_exists:
                print("skip use cache")
                self.load_cache()
                return self.cached_K
        return None
    def load_cache(self):
        self.cached_K = np.load(self.cache_file)

    def save_cache(self):
        np.save(self.cache_file, self.cached_K)
class RBF(CacheKernel):
    def __init__(self, sigma=1., cache_file = ""):
        super().__init__(cache_file)
        self.sigma = sigma  ## the variance of the kernel

    def __call__(self,X,Y, use_cache = True):
        ## Input vectors X and Y of shape Nxd and Mxd
        K = super().__call__(use_cache)
        if K is not None:
            return K
        N, M = (X.shape[0], Y.shape[0])
        X = X.reshape(N, -1)
        Y = Y.reshape(M, -1)
        K = np.zeros((N, M))
        norms = np.linalg.norm(X, axis = 1, keepdims = True)**2 + np.linalg.norm(Y, axis = 1, keepdims = True).T**2 - 2* X @ Y.T
        K = np.exp(-norms/(2*self.sigma**2))
        # K = np.exp(-np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)**2/(2*self.sigma**2)) # N x M
        if use_cache:
            self.cached_K = K
            self.save_cache()

        return K

class Linear(CacheKernel):
    def __call__(self,X,Y, use_cache=True):
        K = super().__call__(use_cache)
        if K is not None:
            return K
        ## Input vectors X and Y of shape Nxd and Mxd
        K = X @ Y.T
        if use_cache:
            self.cached_K = K
            self.save_cache()
        return K ## Matrix of shape NxM
    
class HistogramKernel(CacheKernel):
    def __init__(self, mu, lambd, bins = 16, cache_file = ""):
        super().__init__(cache_file)
        self.mu = mu
        self.lambd = lambd
        self.bins = bins

    def __call__(self, X, Y, use_cache):
        K = super().__call__(use_cache)
        if K is not None:
            return K
        # X: N x h x w x 3
        # Y: M x h x w x 3
        N, M = (X.shape[0], Y.shape[0])
        X = X.reshape(N, -1, 3)
        Y = Y.reshape(M, -1, 3)
        K = np.zeros((N, M))
        for i in tqdm.tqdm(range(N)):
            for j in range(M):   
                p = fast_histogram.histogramdd(X[i], range=[[0, 255], [0, 255], [0, 255]], bins=[self.bins, self.bins, self.bins])
                q = fast_histogram.histogramdd(Y[j], range=[[0, 255], [0, 255], [0, 255]], bins=[self.bins, self.bins, self.bins])     
                # p, _ = np.histogramdd(X[i], bins = self.bins, range = [[0, 255], [0, 255], [0, 255]])
                # q, _ = np.histogramdd(Y[j], bins = self.bins, range = [[0, 255], [0, 255], [0, 255]])
                p = p.flatten()
                p = p/len(p)
                q = q.flatten()
                q = q/len(q)
                eps = 1e-5
                d_ki2 = np.sum((p - q)**2/((p+q)+eps))
                K[i, j] = self.lambd*np.exp(-self.mu*d_ki2)
        if use_cache:
            self.cached_K = K
            self.save_cache()
        return K

class SumKernel(CacheKernel):
    def __init__(self, kernels, weights=None):
        self.kernels = kernels
        self.weights = weights
        if weights is None:
            n = len(self.kernels)
            self.weights = [1/n for _ in range(n)]
    def __call__(self, X, Y, use_cache=True):

        grams = np.array([k(X, Y, use_cache) for k in self.kernels]) # K x N x M
        K = np.average(grams, weights=self.weights, axis = 0) # N x M
        return K
    
    def save(self):
        for k in self.kernels:
            k.save_cache()

    def load(self):
        for k in self.kernels:
            k.load_cache()

kernels_dict = {
    'RBF': RBF,
    'Linear': Linear,
    'HistogramKernel': HistogramKernel
}   
if __name__ == '__main__':
    # Computes the kernel file
    x = np.random.randn(10, 32, 32, 3)
    k = HistogramKernel(2, 1)
    K = k.kernel(x, x)
    print(K.shape, K[0])
