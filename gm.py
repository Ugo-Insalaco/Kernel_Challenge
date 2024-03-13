from data import load_data
from utils import reshape_rescale, get_n_grey_images, compute_sifts
from sklearn.mixture import GaussianMixture
import numpy as np
from joblib import dump

if __name__ == '__main__':
    ncomponents = 128
    n = 0
    max_iter = 200
    h, w = 32, 32
    x_train, y_train, x_test, y_test = load_data(test_size = 0)
    x_train = reshape_rescale(x_train, h, w) # n x 3 x h x w

    x_train_gray = get_n_grey_images(x_train, n)
    x_train_sifts_features = compute_sifts(x_train_gray)
    gm_train = np.concatenate(x_train_sifts_features, axis = 0)
    gm = GaussianMixture(n_components = ncomponents, covariance_type='diag', max_iter=max_iter, verbose=1).fit(gm_train)
    dump(gm, 'models/gm.joblib')