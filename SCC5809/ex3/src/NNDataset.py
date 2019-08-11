'''
Created on Sep 30, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 3: MLP + RBF
@author: Damares Resende
'''

from random import shuffle
import numpy as np
import urllib.request

class NNData():
    
    @staticmethod
    def getSeedsData(normalize = False, binarize = True):
        target_url = 'https://archive.ics.uci.edu/ml/machine-learning-' + \
        'databases/00236/seeds_dataset.txt'
        
        data = urllib.request.urlopen(target_url).read().split(b'\n')
        data_matrix = np.zeros((len(data),len(data[0].split(b'\t')))) 
        
        for idx, line in enumerate(data):
            data_matrix[idx] = [x for x in line.strip().split(b'\t') if x]
        
        X = data_matrix[:,:-1]
        y = data_matrix[:,-1]

        if normalize:
            X = (X - X.min())/(X.max() - X.min())
        if binarize:
            y = NNData.binarizeLabels(y, 3)
            
        return X,  y
    
    @staticmethod
    def binarizeLabels(labels, n):
        targets = np.zeros((labels.shape[0], n))
        
        for i in range(labels.shape[0]):
            targets[i][int(labels[i]) - 1] = 1
        return targets
    
    @staticmethod
    def stratifiedSplit(X, y, test_split):
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
            
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
            