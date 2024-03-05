from svc import KernelSVC
from data import load_data
from kernels import SumKernel, kernels_dict
import numpy as np
import tqdm
import os
from utils import tri_to_list_index, list_to_tri_index, dict_to_filename

class MultiClassSVC():
    def __init__(self,nclasses, C, kernel_name, kernel_kwargs, mode='ovo', epsilon=0.01, weights = None, cache_folder="kernels", cache_prefix=""):
        # Checking mode config
        if mode not in ["ovo", "ovr"]:
            raise ValueError(f"Invalid mode {mode}")
        self._mode = mode

        # Computing number of required kernels
        self.nclasses = nclasses
        if self._mode == "ovr":
            nkernels = nclasses
        elif self._mode == "ovo":
            nkernels = round(nclasses * (nclasses-1)/2)

        # Instanciating the kernels
        kernel_names = kernel_name.split("+")
        cache_files = [{name: os.path.join(cache_folder, f"{name}_{cache_prefix}_{dict_to_filename(kernel_kwargs[name])}_{self._mode}_{k}.npy") for name in kernel_names} for k in range(nkernels)]
        kernels_instances = [[kernels_dict[name](**kernel_kwargs[name], cache_file=cache_files[k][name]) for name in kernel_names] for k in range(nkernels)]
        
        # Sum kernels
        if len(kernel_names) > 1:
            kernel = [SumKernel(kernels_instances[i], weights) for i in range(nkernels)]
        else: 
            kernel = [kernels_instances[i][0] for i in range(nkernels)]

        # Instanciating SVC models
        self.svcs = [KernelSVC(C, k, epsilon=epsilon) for k in kernel]

    @staticmethod
    def split_ovo(Xtr, ytr, i, j):
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
        ds = []
        for k in range(len(self.svcs)):
            print(f"Predicting for classifier {k}")
            ds.append(self.svcs[k].separating_function(X) + self.svcs[k].b)
        ds = np.array(ds)
        # ds = np.array([svc.separating_function(X) + svc.b for svc in self.svcs]) # nclasses x n (ovr) / nclasses*(nclasses-1)/2 x n (ovo)
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
    test_size = 0.1
    x_train, y_train, x_test, y_test = load_data(test_size = test_size)
    # kernel = RBF(sigma=4).kernel
    kernel_kwargs = {
        "HistogramKernel": {
            "mu": 50,
            "lambd": 1
        },
        "RBF":{
            "sigma": 4
        },
        "Linear": {}
    }
    kernel_name = "RBF"
    # kernel = kernels_dict[kernel_name](kernel_kwargs[kernel_name]).kernel
    classifier = MultiClassSVC(10, 1e1, kernel_name, kernel_kwargs, 'ovo', epsilon = 1e-5, cache_prefix=f"s{test_size}")
    # classifier = MultiClassSVC(10, 25, Linear().kernel, 'ovo', epsilon = 1e-10) # test_size=0.15
    classifier.fit(x_train,y_train)
    classifier.save('models/multiclass_svc_rbf')