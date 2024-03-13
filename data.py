import numpy as np
import os
from image_viewer import ImageViewer
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(test_size = .2, use_fisher_vectors=False):
    print("== Loading data ==")
    np.random.seed(123456789)
    data_path = 'data'
    if not use_fisher_vectors:
        x_train_path = os.path.join(data_path, 'Xtr.csv')
        y_train_path = os.path.join(data_path, 'Ytr.csv')
        x_test_path = os.path.join(data_path, 'Xte.csv')
        Xtr = np.array(pd.read_csv(x_train_path,header=None,sep=',',usecols=range(3072))) 
        Ytr = np.array(pd.read_csv(y_train_path,sep=',',usecols=[1])).squeeze() 
        if test_size == 0:
            Xte, Yte = np.array(pd.read_csv(x_test_path,header=None,sep=',',usecols=range(3072))) , None
        else:
            Xtr, Xte, Ytr, Yte = train_test_split(Xtr, Ytr, test_size=test_size)
    else:
        x_train_path = os.path.join(data_path, 'Xtr_fisher.npy')
        y_train_path = os.path.join(data_path, 'Ytr.csv')
        x_test_path = os.path.join(data_path, 'Xte_fisher.npy')
        Xtr = np.load(x_train_path)
        Ytr = np.array(pd.read_csv(y_train_path,sep=',',usecols=[1])).squeeze()
        if test_size == 0:
            Xte, Yte = np.load(x_test_path), None
        else:
            Xtr, Xte, Ytr, Yte = train_test_split(Xtr, Ytr, test_size=test_size)
    return Xtr, Ytr, Xte, Yte

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data(test_size = 0.2)
    w, h = 32, 32
    print(x_train, y_train)
    x_train = np.reshape(x_train, (-1, 3, h, w))
    x_train = np.moveaxis(x_train, 1, 3) # N x h x w x 3
    print(x_train.shape, y_train.shape)
    # viewer = ImageViewer(x_train, y_train)

    x_train, y_train, x_test, y_test = load_data(test_size = 0.2, use_fisher_vectors=True)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)