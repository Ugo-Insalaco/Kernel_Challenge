#################################################################################################################################
####### Kernel SVC Class, completed from https://mva-kernel-methods.github.io/course-2023-2024/assignments/02_assignment ########
#################################################################################################################################

import numpy as np
from scipy import optimize

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
        N = len(y)
        self.X = X
        self.y = y
        dy = np.diag(y)
        K = self.kernel(X, X)
        # Lagrange dual problem
        def loss(alpha):
            l = -np.sum(alpha)+ 1/2 * alpha.T @ dy @ K @ dy @ alpha
            return l #'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            g = -np.ones(N)+ dy @ K @ dy @ alpha
            return g# '''----------------partial derivative of the dual loss wrt alpha -----------------'''
            
        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
        
        fun_eq = lambda alpha: np.dot(y, alpha) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha: y #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha: np.concatenate((np.ones(N)*self.C - alpha, alpha))#  2N'''---------------function defining the inequality constraint-------------------'''  
        jac_ineq = lambda alpha:np.concatenate((-1*np.identity(N), np.identity(N)))  # 2N x N# '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''

        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq','fun': fun_ineq, 'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x
        
        ## Assign the required attributes
        print(self.alpha)
        self.alpha[self.alpha < self.epsilon] = 0
        self.alpha[self.alpha > self.C - self.epsilon] = self.C
        condsupp = (self.alpha>0) & (self.alpha < self.C)
        self.support = X[condsupp] #'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        fsupp = self.separating_function(self.support)
        ysupp= y[condsupp]
        self.b = ((1/ysupp-fsupp)).mean() #''' 0]-----------------offset of the classifier------------------ ''' # k(N x d, 1 x d) = N x 1
        self.norm_f = np.sum(self.alpha**2)# '''------------------------RKHS norm of the function f ------------------------------'''

    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        ka = self.kernel(self.X,x)*self.alpha[:, None]*self.y[:, None] #  k(M x d, N x d) = M x N 
        f = np.sum(ka, axis = 0) # N
        return f
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1