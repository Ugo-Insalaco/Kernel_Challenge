import numpy as np 
from joblib import load
from utils import MVN, reshape_rescale, compute_sifts, get_n_grey_images, filter_gm
from data import load_data
import tqdm


def gammas(x,alpha,mvns):
    # x: T x d
    # alpha: d
    T=x.shape[0]
    p=alpha.shape[0]
    u=np.zeros((p, T))
    for k in range(p):
        u[k]=mvns[k](x)
    gamma_vect=alpha[:, None]*u
    # print(alpha, 'u', u, 'gamma', gamma_vect)
    return gamma_vect/np.sum(gamma_vect, axis = 0) # p x T

def fisher_vector(descriptors,alpha,mu,sigma):
    # descriptors: d x T
    # alpha: d
    # mu: p x d
    # sigma: p x d
    mvns = [MVN(mu[k], sigma[k]) for k in range(len(mu))]
    d,T= descriptors.shape
    p=alpha.shape[0]
    #compute statistics
    S0=np.zeros(p)
    S1=np.zeros((p,d))
    S2=np.zeros((p,d))

    gamma_vect = gammas(descriptors.T, alpha,mvns) # p x T
    S0 = np.sum(gamma_vect, axis = 1) # p
    s1 = gamma_vect[:, None, :] * descriptors[None, : , :] # p x 1 x T * 1 x d x T = p x d x T
    S1 = np.sum(s1, axis = 2) # p x d
    s2 = gamma_vect[:, None, :] * descriptors[None, : , :]**2 # p x 1 x T * 1 x d x T = p x d x T
    S2 = np.sum(s2, axis = 2) # p x d

    #fisher vector signature
    G_alpha=(S0-T*alpha)/np.sqrt(alpha)
    G_mu=(S1-mu*S0[:, None])/(np.sqrt(alpha[:, None])*sigma) # (p x d - p x d * p x 1) / (p x 1 * p x d) = p x d
    G_sigma=(S2-2*mu*S1+(mu**2-sigma**2)*S0[:, None])/(np.sqrt(2*alpha[:, None])*sigma**2)
    G_lambda=np.concatenate((G_alpha,G_mu.flatten(),G_sigma.flatten()))
    
    #normalization
    G_lambda=np.sign(G_lambda)*np.sqrt(np.abs(G_lambda))
    G_lambda=G_lambda/np.sqrt(G_lambda.T@G_lambda)
    return G_lambda

def fisher_vectors_matrix(sifts_features, gm):
    #X of size N*d*d
    alpha, mu, sigma = gm.weights_, gm.means_, gm.covariances_
    alpha, mu, sigma = filter_gm(alpha, mu, sigma, 1/len(alpha))

    M=[]
    for k in tqdm.tqdm(range(len(sifts_features))):
        fish_vector= fisher_vector(sifts_features[k].T,alpha,mu,sigma)
        M.append(fish_vector)
    M = np.stack(M, axis = 0)
    return M


if __name__ == '__main__':
    h, w = 32, 32
    x_train, y_train, x_test, y_test = load_data(test_size = 0)
    x_test = reshape_rescale(x_test, h, w) # n x 3 x h x w
    x_train = reshape_rescale(x_train, h, w)

    gm = load('models/gm.joblib')

    x_test_gray = get_n_grey_images(x_test)
    x_test_sifts_features = compute_sifts(x_test_gray)
    fv_test = fisher_vectors_matrix(x_test_sifts_features, gm)
    np.save('data/Xte_fisher.npy', fv_test)

    x_train_gray = get_n_grey_images(x_train)
    x_train_sifts_features = compute_sifts(x_train_gray)
    fv_train = fisher_vectors_matrix(x_train_sifts_features, gm)
    np.save('data/Xtr_fisher.npy', fv_train)

