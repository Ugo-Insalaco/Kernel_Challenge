from sklearn import metrics
from data import load_data
from multiclass_svc import MultiClassSVC
from kernels import Linear, RBF, HistogramKernel
import pandas as pd
import os

def compute_metrics(classifier, x_test, y_test, model_name):
    path = 'results'
    pred_path = os.path.join(path, f'y_pred_{model_name}.csv')
    metric_path = os.path.join(path, f'metrics_{model_name}.csv')

    if os.path.isfile(pred_path):
        print("Existing file {pred_path}, skipping evaluation")
        y_pred = pd.read_csv(pred_path)["Prediction"].to_numpy()
    else:
        y_pred = classifier.predict(x_test) 
        y_pred_dict = {'Prediction' : y_pred} 
        dataframe = pd.DataFrame(y_pred_dict) 
        dataframe.index += 1 
        dataframe.to_csv(pred_path,index_label='Id') 
    result_metrics = {
               'precision': lambda yt, yp: metrics.precision_score(yt, yp, average=None), 
               'recall': lambda yt, yp: metrics.recall_score(yt, yp, average=None), 
               'f1': lambda yt, yp: metrics.f1_score(yt, yp, average=None), 
               'accuracy': lambda yt, yp: metrics.accuracy_score(yt, yp)
               }
    
    for metric in result_metrics:
        result_metrics[metric] = result_metrics[metric](y_test, y_pred)
    
    with open(metric_path, 'w') as f:
        for key in result_metrics.keys():
            f.write("%s,%s\n"%(key,result_metrics[key]))

if __name__ == '__main__':
    test_size = 0.1
    x_train, y_train, x_test, y_test = load_data(test_size = test_size)
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
    kernel_name = "HistogramKernel+RBF"
    classifier = MultiClassSVC(10, 1e1, kernel_name, kernel_kwargs, 'ovo', epsilon = 1e-5, cache_prefix=f"s{test_size}")
    # classifier = MultiClassSVC(10, 100, Linear().kernel, epsilon = 1e-10)
    # classifier = MultiClassSVC(10, 1e1, RBF(sigma=4).kernel, 'ovo', epsilon = 1e-16)
    classifier.load('models/multiclass_svc_hist+rbf.npz', x_train, y_train)
    
    compute_metrics(classifier, x_test, y_test, 'multiclass_svc_hist+rbf')