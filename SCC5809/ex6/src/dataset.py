'''
Created on Nov 25, 2018

@author: damaresresende
'''
from __future__ import print_function


import pandas as pd


class Data:
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
        
        data = df.loc[:, features].values
        labels = df.loc[:,['target']].values
    
        return data, labels
