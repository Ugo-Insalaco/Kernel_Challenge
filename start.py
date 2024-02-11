from data import load_data
import pandas as pd
if __name__ == '__main__':
    x_train, y_train, x_test = load_data()
    classifier = None
    classifier.train(x_train,y_train)
    Yte = classifier.fit(x_test) 
    Yte = {'Prediction' : Yte} 
    dataframe = pd.DataFrame(Yte) 
    dataframe.index += 1 
    dataframe.to_csv('Yte_pred.csv',index_label='Id') 