from math import *
from scipy import optimize
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import numpy as np
import cv2

def list_to_tri_index(k, n):
    # i < j
    i = n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    return i, j

def tri_to_list_index(i, j, n):
    # i < j
    k = round((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
    return k

def dict_to_filename(dict):
    return "_".join([f"{key}_{dict[key]}" for key in dict])

def solve_cvxopt(K, dy, C, y):
    N = K.shape[0]
    
    Q = matrix(dy @ K @ dy)
    p = matrix(-np.ones(N))
    G = matrix(np.concatenate((-1*np.identity(N), np.identity(N))))
    h = matrix(np.concatenate((np.zeros(N), np.ones(N)*C)))
    A = matrix(y.astype('float'), (1, N))
    b = matrix(0., (1, 1))
    
    sol = solvers.qp(Q, p, G, h, A, b)
    return np.array(sol['x'])[:, 0]

def solve_scipy(K, dy, C, y):
    N = K.shape[0]
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
    fun_ineq = lambda alpha: np.concatenate((np.ones(N)*C - alpha, alpha))#  2N'''---------------function defining the inequality constraint-------------------'''  
    jac_ineq = lambda alpha:np.concatenate((-1*np.identity(N), np.identity(N)))  # 2N x N# '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''

    
    constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                    {'type': 'ineq','fun': fun_ineq, 'jac': jac_ineq})
    optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                x0=np.ones(N), 
                                method='SLSQP', 
                                jac=lambda alpha: grad_loss(alpha), 
                                constraints=constraints)
    return optRes.x

class MVN():
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.det_cov = np.prod(self.cov)
        self.inv_cov = 1/(self.cov) # d
        self.d = mean.shape[0]
    def __call__(self, x):
        # x: T x d
        # return T
        xm = x - self.mean[None, :] # T x d - 1 x d
        dot = xm ** 2 * self.inv_cov[None, :] # T x d * 1 x d = T x d
        dens = np.sqrt(2*np.pi*self.cov) # d
        prod = 1/dens[None, :]*np.exp(-1/2*dot) # T x d

        return np.prod(prod, axis = 1)
    
def compute_sifts(X):
    sifts_features = []
    i=0
    for image in X:
        i+=1
        sift = cv2.SIFT_create(nfeatures=256, contrastThreshold  = 1e-3, edgeThreshold = 20, sigma = 0.8)
        _, descriptors = sift.detectAndCompute(image, None) # T x d
        descriptors=descriptors/np.sum(descriptors, axis = 1, keepdims=True)
        sifts_features.append(descriptors)
    return sifts_features

def get_n_grey_images(x, n=0):
    grays = []
    if n == 0:
        n = x.shape[0]
        
    for i in range(n):
        img = x[i]*256
        img = img.astype(np.uint8)
        img = np.moveaxis(img, 0, 2) # h x w x 3 -> 3 x w x h
        img = np.moveaxis(img, 1, 2) # 3 x w x h -> 3 x h x w
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        grays.append(gray)
    return grays

def reshape_rescale(x, h, w):
    n = x.shape[0]
    x = np.reshape(x, (n, 3, h, w))
    x = np.moveaxis(x, 1, 3)
    x = 0.5*(x+1)
    return x

def filter_gm(alpha, mu, sigma, thresh):
    keep = alpha > thresh
    print(f'Removing {np.sum(1-keep)} gaussians')
    new_alpha = alpha[keep]
    new_alpha = new_alpha/np.sum(new_alpha)
    new_mu = mu[keep]
    new_sigma = sigma[keep]
    return new_alpha, new_mu, new_sigma
