import numpy as np
import os
from image_viewer import ImageViewer
import pandas as pd
def load_data():
    data_path = 'data'
    x_train_path = os.path.join(data_path, 'Xtr.csv')
    y_train_path = os.path.join(data_path, 'Ytr.csv')
    x_test_path = os.path.join(data_path, 'Xte.csv')
    Xtr = np.array(pd.read_csv(x_train_path,header=None,sep=',',usecols=range(3072))) 
    Xte = np.array(pd.read_csv(x_test_path,header=None,sep=',',usecols=range(3072))) 
    Ytr = np.array(pd.read_csv(y_train_path,sep=',',usecols=[1])).squeeze() 
    return Xtr, Ytr, Xte

if __name__ == "__main__":
    x_train, y_train, x_test = load_data()
    w, h = 32, 32
    print(x_train, y_train)
    x_train = np.reshape(x_train, (-1, 3, h, w))
    x_train = np.moveaxis(x_train, 1, 3) # N x h x w x 3
    print(y_train)
    viewer = ImageViewer(x_train, y_train)