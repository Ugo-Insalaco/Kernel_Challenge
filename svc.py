#################################################################################################################################
####### Kernel SVC Class, completed from https://mva-kernel-methods.github.io/course-2023-2024/assignments/02_assignment ########
#################################################################################################################################

import numpy as np
from utils import solve_cvxopt, solve_scipy

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.b=None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
    
        self.X = X
        self.y = y
        dy = np.diag(y)
        print("Computing kernel")
        K = self.kernel(X, X, use_cache=True)
        print("Done computing kernel")
        print("Starting optimization")
        self.alpha = solve_cvxopt(K, dy, self.C, y)

        print("Done optimizing")
        ## Assign the required attributes
        # print(self.alpha)
        self.alpha[self.alpha <= self.epsilon] = 0
        self.alpha[self.alpha >= self.C - self.epsilon] = self.C
        condsupp = (self.alpha>0) & (self.alpha < self.C)
        self.support = X[condsupp] #'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        if len(self.support) == 0:
            print("warning: empty support")
            self.b = 0
        else:
            print(f"found {len(self.support)} supports")
            fsupp = self.separating_function(self.support)
            ysupp= y[condsupp]
            self.b = ((1/ysupp-fsupp)).mean() #''' 0]-----------------offset of the classifier------------------ ''' # k(N x d, 1 x d) = N x 1
            self.norm_f = np.sum(self.alpha**2)# '''------------------------RKHS norm of the function f ------------------------------'''

    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        ka = self.kernel(self.X,x, use_cache=False)*self.alpha[:, None]*self.y[:, None] #  k(M x d, N x d) = M x N 
        f = np.sum(ka, axis = 0) # N
        return f
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1