import numpy as np 
from joblib import load
from utils import MVN, reshape_rescale, compute_sifts, get_n_grey_images, filter_gm, compute_hogs
from data import load_data
import tqdm
from skimage.feature import fisher_vector as fv_sklearn

def fisher_vector(descriptors,gm):
    # descriptors: T x d
    # alpha: d
    # mu: p x d
    # sigma: p x d
    alpha, mu, sigma2 = gm.weights_, gm.means_, gm.covariances_

    T, d= descriptors.shape
    p=alpha.shape[0]

    #compute statistics
    gamma_vect = gm.predict_proba(descriptors).T # p x T
    S0 = np.mean(gamma_vect, axis = 1) # p
    S1 = gamma_vect @ descriptors / T # p x T @ T x d = p x d
    S2 = gamma_vect @ descriptors**2 # p x 1 x T * 1 x d x T = p x d x T

    #fisher vector signature
    G_alpha=(S0 - alpha) / np.sqrt(alpha)
    G_mu=(S1 - mu*S0[:, None]) / (np.sqrt(alpha[:, None]) * np.sqrt(sigma2)) # (p x d - p x d * p x 1) / (p x 1 * p x d) = p x d
    G_sigma=-(S2 - 2 * mu * S1 + (mu**2 - sigma2) * S0[:, None]) / (np.sqrt(2 * alpha[:, None]) * sigma2)
    G_lambda=np.concatenate((G_alpha,G_mu.flatten(),G_sigma.flatten()))

    #normalization
    G_lambda=np.sign(G_lambda)*np.sqrt(np.abs(G_lambda))
    G_lambda=G_lambda/np.sqrt(G_lambda.T@G_lambda)
    return G_lambda

def fisher_vectors_matrix(features, gm):
    #X of size N*T*d
    # alpha, mu, sigma = filter_gm(alpha, mu, sigma, 1/len(alpha))

    M=[]
    for k in tqdm.tqdm(range(len(features))):
        fish_vector= fisher_vector(features[k],gm)
        M.append(fish_vector)
    M = np.stack(M, axis = 0)
    return M


if __name__ == '__main__':
    feature_extraction = 'hog'
    h, w = 32, 32
    x_train, y_train, x_test, y_test = load_data(test_size = 0)
    x_test = reshape_rescale(x_test, h, w) # n x 3 x h x w
    x_train = reshape_rescale(x_train, h, w)

    gm = load('models/gm.joblib')

    if feature_extraction == 'sift':
        x_test_gray = get_n_grey_images(x_test)
        x_test_features = compute_sifts(x_test_gray)
        x_train_gray = get_n_grey_images(x_train)
        x_train_features = compute_sifts(x_train_gray)
    elif feature_extraction == 'hog':
        x_test_features = compute_hogs(x_test)
        x_train_features = compute_hogs(x_train)
    else:
        raise ValueError(f'Unknown feature extracton {feature_extraction}')
    
    fv_test = fisher_vectors_matrix(x_test_features, gm)
    fv_train = fisher_vectors_matrix(x_train_features, gm)
    np.save('data/Xte_fisher.npy', fv_test)
    np.save('data/Xtr_fisher.npy', fv_train)

