'''
Created on Nov 21, 2018

@author: damaresresende
'''
import pandas as pd
import numpy as np
from random import shuffle

class NNData():    
    @staticmethod
    def get_wine_data():
        url = "https://archive.ics.uci.edu/ml/machine-" + \
                "learning-databases/wine/wine.data"
                
        df = pd.read_csv(url, names=['target', 'Alcohol', 'Malic acid', 'Ash', 
                    'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 
                    'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 
                    'Hue','OD280/OD315 of diluted wines', 'Proline'])
        
        features = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
                    'Magnesium', 'Total phenols', 'Flavanoids', 
                    'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 
                    'Hue', 'OD280/OD315 of diluted wines', 'Proline']
        
        x = df.loc[:, features].values
        y = df.loc[:,['target']].values
    
        return x, y - 1
    
    @staticmethod
    def normalize(x):
        return (x - x.min())/(x.max() - x.min())
    
    @staticmethod
    def standardize(x):
        return (x - np.mean(x)) / np.std(x)
        
    @staticmethod
    def binarize_labels(labels, n):
        targets = np.zeros((labels.shape[0], n))
        
        for i in range(labels.shape[0]):
            targets[i][int(labels[i]) - 1] = 1
        return targets
    
    @staticmethod
    def stratified_split(X, y, test_split):
        data_map = {}
        
        # split dataset according to each class
        for idx, ex in enumerate(X):
            if str(y[idx]) not in data_map.keys():
                data_map[str(y[idx])] = [(ex, y[idx])]
            else:
                data_map[str(y[idx])].append((ex, y[idx]))
        
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        
        # shuffle the lists and get a share of each class
        for set_ in data_map.values():
            shuffle(set_)
            limit = round(test_split * len(set_))
            
            X_test.extend([ex[0] for ex in set_[:limit]])
            y_test.extend([ex[1] for ex in set_[:limit]])
            X_train.extend([ex[0] for ex in set_[limit:]])
            y_train.extend([ex[1] for ex in set_[limit:]])
            
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)