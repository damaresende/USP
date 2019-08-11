'''
Created on Oct 25, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project II: CNN
@author: Damares Resende

Just stores the parameters used to train the CNN.
'''

import os

class CNNParams:
    n_epochs = 50
    n_classes = 8
    img_height = 96
    img_width = 96
    n_channels = 1
    batch_size = 32
    model_shape = {'conv': {'filters': [32, 16, 24], 
                            'kernel': [(3,3), (3,3), (3,3)], 
                            'pooling': [True, False, True],
			    'dropout': [0, 0.3, 0.1]}, 
                   'dense': {'size': [24, 16],
			     'dropout': [0.4, 0.1]}}
    model_path = os.path.join(os.getcwd(), 'log/qider_model_m2.h5')
