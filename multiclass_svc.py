from svc import KernelSVC
from data import load_data
from kernels import Linear, RBF
import numpy as np
import tqdm
from utils import tri_to_list_index, list_to_tri_index

class MultiClassSVC():
    def __init__(self,nclasses, C, kernel , mode='ovo', epsilon=0.01):
        self.nclasses = nclasses
        if mode not in ["ovo", "ovr"]:
            raise ValueError(f"Invalid mode {mode}")
        self._mode = mode
        if self._mode == "ovr":
            self.svcs = [KernelSVC(C, kernel, epsilon=epsilon) for _ in range(nclasses)] # SVC i classifies i against all others
        elif self._mode == "ovo":
            self.svcs = [KernelSVC(C, kernel, epsilon=epsilon) for _ in range(round(nclasses * (nclasses-1)/2))]
        self._mode = mode

    def split_ovo(self, Xtr, ytr, i, j):
        new_data = Xtr[(ytr==i) | (ytr==j)]
        new_target = ytr[(ytr==i) | (ytr==j)]
        new_target[new_target == i] = 1 # i is 1
        new_target[new_target == j] = -1 # j is -1
        return new_data, new_target
    
    def fit(self, X, y):
        print("== Fitting MultiSVC model ==")
        # X: n x d
        n = X.shape[0]
        if self._mode == "ovr":
            for i in tqdm.tqdm(range(self.nclasses)):
                new_target = np.zeros(n) # n
                target_indexes = np.arange(n)[y == i] # n
                against_indexes = np.arange(n)[y != i] # n
                new_target[target_indexes] = 1
                new_target[against_indexes] = -1
                self.svcs[i].fit(X, new_target)
        elif self._mode == "ovo":
            for i in range(self.nclasses):
                for j in range(i+1, self.nclasses):
                    print(f"==== Fitting Class {i} against {j} ====")
                    new_data, new_target = self.split_ovo(X, y, i, j)
                    k = tri_to_list_index(i, j, self.nclasses)
                    self.svcs[k].fit(new_data, new_target)
    
    def save(self, file):
        save_data = [np.append(svc.alpha, svc.b) for svc in self.svcs]
        save_data = {f'alpha_{i}':save_data[i] for i in range(len(save_data))}
        np.savez(file, **save_data)

    def load(self, file, x_train, y_train):
        save_data = np.load(file, allow_pickle=True)
        save_data = [save_data[key] for key in list(save_data)]
        alphas = [save_data[i][:-1] for i in range(len(save_data))]
        bs = [save_data[i][-1] for i in range(len(save_data))]

        for i in range(len(self.svcs)):
            self.svcs[i].alpha = alphas[i]
            self.svcs[i].b = bs[i]

        self.set_support(x_train, y_train)
            
    def set_support(self, Xtr, ytr):
        if self._mode == 'ovr':
            for svc in self.svcs:
                svc.X = Xtr
                svc.y = ytr
        elif self._mode == 'ovo':
            for i in range(self.nclasses):
                for j in range(i+1, self.nclasses):
                    new_data, new_target = self.split_ovo(Xtr, ytr, i, j)
                    k = tri_to_list_index(i, j, self.nclasses)
                    self.svcs[k].X = new_data
                    self.svcs[k].y = new_target
    
    def predict(self, X):
        # X: n x d
        n = X.shape[0]
        ds = np.array([svc.separating_function(X) + svc.b for svc in self.svcs]) # nclasses x n (ovr) / nclasses*(nclasses-1)/2 x n (ovo)
        if self._mode == "ovr":
            ypred = np.argmax(ds, axis = 0) # n
        elif self._mode == "ovo":
            votes = np.zeros((n, self.nclasses))
            for k in range(round(self.nclasses * (self.nclasses - 1)/2)):
                i, j = list_to_tri_index(k, self.nclasses)
                p = 2 * (ds[k]>0) - 1 # n
                p[p == 1] = i
                p[p == -1] = j
                votes[np.arange(n), p]+=1
            return np.argmax(votes, axis = 1)


        return ypred

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(test_size = 0.15)
    classifier = MultiClassSVC(10, 1e1, RBF(sigma=25).kernel, 'ovo', epsilon = 1e-16)
    # classifier = MultiClassSVC(10, 25, Linear().kernel, 'ovo', epsilon = 1e-10) # test_size=0.15
    classifier.fit(x_train,y_train)
    classifier.save('models/multiclass_svc_RBF')