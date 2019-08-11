'''
Created on Nov 4, 2018

@author: damaresresende
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCAWine:
    @staticmethod
    def get_data():
        url = "https://archive.ics.uci.edu/ml/machine-" + \
                "learning-databases/wine/wine.data"
                
        df = pd.read_csv(url, names=['target', 'Alcohol', 'Malic acid', 'Ash', 
                             'Alcalinity of ash', 'Magnesium', 'Total phenols',
                             'Flavanoids', 'Nonflavanoid phenols', 
                             'Proanthocyanins', 'Color intensity', 'Hue',
                             'OD280/OD315 of diluted wines', 'Proline'])
        
        features = ['Alcohol', 'Malic acid', 'Ash', 
            'Alcalinity of ash', 'Magnesium', 'Total phenols',
            'Flavanoids', 'Nonflavanoid phenols', 
            'Proanthocyanins', 'Color intensity', 'Hue',
            'OD280/OD315 of diluted wines', 'Proline']
        
        x = df.loc[:, features].values
        y = df.loc[:,['target']].values
        
        return x, y
    
    @staticmethod
    def standardize_data(x):
        return (x - np.mean(x)) / np.std(x)
    
    @staticmethod
    def calc_pca(data, dims_rescaled_data=2):
        # center the data
        data = data.mean(axis=0) - data
        # calc covariance matrix
        cov_mat = np.cov(data, rowvar=False)
        # calc eigenvectors and eigenvalues
        evals, evecs = np.linalg.eig(cov_mat)
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first n eigenvectors
        evecs = evecs[:, :dims_rescaled_data]
        # return rescaled data, eigenvalues, and eigenvectors
        return np.dot(data, evecs), evals, evecs
    
    @staticmethod
    def get_variance_ratio(evals):
        return abs(evals / np.sum(evals))
    
    @staticmethod
    def plot_data(x, y):
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Componente Principal 1', fontsize = 15)
        ax.set_ylabel('Componente Principal 2', fontsize = 15)
        ax.set_title('PCA com 2 componentes', fontsize = 20)
        targets = [1,2,3]
        colors = ['r', 'g', 'b']
        
        for target, color in zip(targets,colors):
            indicesToKeep = y == target
            ax.scatter(x[:,0][indicesToKeep[:,0]]
                       , x[:,1][indicesToKeep[:,0]]
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()
        
        plt.show()

x, y = PCAWine.get_data()
std_x = PCAWine.standardize_data(x)

print('Results of the coded algorithm...')
components, evals, _ = PCAWine.calc_pca(std_x, 2)
var_ratio = PCAWine.get_variance_ratio(evals)
PCAWine.plot_data(components, y)
print('Variance ratio: %s' % str([round(v, 4) for v in var_ratio]))

print('Comparing with StandardScaler from sklearn...')
from sklearn.preprocessing import StandardScaler
components, evals, _ = PCAWine.calc_pca(StandardScaler().fit_transform(x), 2)
var_ratio = PCAWine.get_variance_ratio(evals)
PCAWine.plot_data(components, y)
print('Variance ratio: %s' % str([round(v, 4) for v in var_ratio]))

print('bye')
