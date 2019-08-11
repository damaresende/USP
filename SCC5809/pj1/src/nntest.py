'''
Created on Sep 8, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project I: MLP
@author: Damares Resende
'''

import os
import enum
import numpy as np
from math import sqrt

# modules used for reading the dataset and spliting it in a stratifid way
import pandas as pd
from sklearn.model_selection import train_test_split #TODO: build my own function for this

class NNType(enum.Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

class NNData():
    
    def get_dummy_data(self):
        X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
 
        y = np.array([[0],
                      [1],
                      [1],
                      [0]])
        return X, y
        
    def get_wine_data(self, test_size):
        data = pd.read_csv(os.getcwd() + '/datasets/wine.csv')
        y = data['0'] 
        X = data.ix[:, '1':]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        
        y_train = y_train.values
        targets_train = np.zeros((y_train.shape[0], 3))
        for i in range(y_train.shape[0]):
            targets_train[i][y_train[i] - 1] = 1
            
        y_test = y_test.values
        targets_test = np.zeros((y_test.shape[0], 3))
        for i in range(y_test.shape[0]):
            targets_test[i][y_test[i] - 1] = 1
        
        return X_train.values, X_test.values, targets_train, targets_test
    
    def get_music_data(self, test_size):
        data = pd.read_csv(os.getcwd() + '/datasets/default_features_1059_tracks.csv')
        y = data.ix[:,'68':'69'] 
        X = data.ix[:, :'67']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        
        return X_train.values, X_test.values, y_train.values, y_test.values

class NeuralNet():
    
    def __init__(self, nn_inputs, nn_hidden, nn_hidden2, nn_targets, nntype):
        self.nn_inputs = nn_inputs
        self.nn_hidden = nn_hidden
        self.nn_targets = nn_targets
        self.nntype = nntype
            
        self.weights0 = np.random.uniform(-1, 1, (nn_inputs, nn_hidden)) 
        self.weights1 = np.random.uniform(-1, 1, (nn_hidden, nn_hidden2))
        self.weights2 = np.random.uniform(-1, 1, (nn_hidden2, nn_targets))
    
    def feedforward(self, X):
        self.a1 = self.activation_fn(np.dot(X, self.weights0))
        self.a2 = self.activation_fn(np.dot(self.a1, self.weights1))
        self.y_hat = self.activation_fn(np.dot(self.a2, self.weights2))
    
    def backpropagation(self, X, y, l_rate, momentum):
        delty_hat = 2 * (y - self.y_hat) * self.activation_fn_derivative(self.y_hat)
        delta2 = np.dot(delty_hat, self.weights2.T) * self.activation_fn_derivative(self.a2)
        delta1 = np.dot(delta2, self.weights1.T) * self.activation_fn_derivative(self.a1)
        
        d_weights2 = l_rate * np.dot(self.a2.T, delty_hat)
        d_weights1 = l_rate * np.dot(self.a1.T, delta2)
        d_weights0 = l_rate * np.dot(X.T, delta1)
        
        self.weights0 += momentum * d_weights0
        self.weights1 += momentum * d_weights1
        self.weights2 += momentum * d_weights2
        
    def train(self, X, y, n_epochs, l_rate, momentum):
        for ep in range(n_epochs):
            self.feedforward(X)
            error = y - self.y_hat
            
            if ep % 1000 == 0:
                print('Error: ' + str(np.mean(np.abs(error))))
            
            self.backpropagation(X, y, l_rate, momentum)
        
    def activation_fn(self, x):
        return 1 / (1 + np.exp(-x))
    
    def activation_fn_derivative(self, y_hat):
        return y_hat * (1 - y_hat)
    
    def predict(self, X):
        self.feedforward(X)
        if self.nntype == NNType.CLASSIFICATION:
            return np.argmax(self.y_hat, axis=1)
        
        return self.y_hat
    
    def calc_accuracy(self, y, y_hat):
        errors = 0
        for i in range(len(y)):
            if  y[i][y_hat[i]] != 1:
                errors += 1
        return 100 - (errors * 100 / len(y))
    
    def calc_rmse(self, y, y_hat):
        error = []
        for err in sum((y - y_hat) * (y - y_hat)):
            error.append(sqrt(err) / len(y))
        return sum(error) / len(y[0])
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        if self.nntype == NNType.CLASSIFICATION:
            return self.calc_accuracy(y, y_hat)
        return self.calc_rmse(y, y_hat)

# Get Data
X_train, X_test, y_train, y_test = NNData().get_wine_data(.2)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())

# Define Parameters
nn_inputs = X_train.shape[1]
nn_hidden1 = 20
nn_hidden2 = 10
nn_targets = y_train.shape[1]
        
n_epochs = 20000
l_rate = 0.3
momentum = 0.75

# Train neural network
nn = NeuralNet(nn_inputs, nn_hidden1, nn_hidden2, nn_targets, NNType.CLASSIFICATION)
nn.train(X_train, y_train, n_epochs, l_rate, momentum)
print('\nAccuracy in training: ' + str(round(nn.evaluate(X_train, y_train),3)) + ' %')
print('Accuracy in testing: ' + str(round(nn.evaluate(X_test, y_test),3)) + ' %')
