from data import load_data
from utils import reshape_rescale, get_n_grey_images, compute_sifts, compute_hogs
from sklearn.mixture import GaussianMixture
import numpy as np
from joblib import dump

if __name__ == '__main__':
    feature_extraction = 'hog'
    ncomponents = 16
    n = 0
    max_iter = 200
    h, w = 32, 32
    x_train, y_train, x_test, y_test = load_data(test_size = 0.1)
    x_train = reshape_rescale(x_train, h, w) # n x h x w x 3

    if feature_extraction == 'sift':
        x_train_gray = get_n_grey_images(x_train, n)
        x_train_features = compute_sifts(x_train_gray)
        gm_train = np.concatenate(x_train_features, axis = 0)
    elif feature_extraction == 'hog':
        x_train_features = compute_hogs(x_train)
        gm_train = np.concatenate(x_train_features, axis = 0)
    else:
        raise ValueError(f'Unknown feature extracton {feature_extraction}')
    gm = GaussianMixture(n_components = ncomponents, covariance_type='diag', max_iter=max_iter, verbose=1, random_state=8).fit(gm_train)
    dump(gm, 'models/gm.joblib')