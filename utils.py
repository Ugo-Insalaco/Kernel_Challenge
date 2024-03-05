from math import *
from scipy import optimize
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import numpy as np

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