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

import matplotlib.pyplot as plt

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
    
    def __init__(self, nn_inputs, nn_hidden, nn_targets, nntype):
        self.nn_inputs = nn_inputs
        self.nn_hidden = nn_hidden
        self.nn_targets = nn_targets
        self.nntype = nntype
            
        self.weights1 = np.random.uniform(-1, 1, (nn_inputs, nn_hidden)) 
        self.weights2 = np.random.uniform(-1, 1, (nn_hidden, nn_targets))
    
    def feedforward(self, X):
        self.a1 = self.activation_fn(np.dot(X, self.weights1))
        self.y_hat = self.activation_fn(np.dot(self.a1, self.weights2))
    
    def backpropagation(self, X, y, l_rate, momentum):
        delta = 2 * (y - self.y_hat) * self.activation_fn_derivative(self.y_hat)
        
        d_weights2 = l_rate * np.dot(self.a1.T, delta)
        d_weights1 = np.dot(X.T, (np.dot(delta, self.weights2.T) * self.activation_fn_derivative(self.a1)))
        
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
X_train, X_test, y_train, y_test = NNData().get_music_data(.2)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())

# Define Parameters
nn_inputs = X_train.shape[1]
nn_hidden = 15
nn_targets = y_train.shape[1]
        
n_epochs = 10000
l_rate = 0.3
momentum = 0.95

test_set_size= [.1, .2, .3, .4, .5]
results_train = np.zeros(len(test_set_size))
results_test = np.zeros(len(test_set_size))
for i in range(len(test_set_size)):
    # Get Data
    X_train, X_test, y_train, y_test = NNData().get_music_data(test_set_size[i])
    X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
    X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())
     
    # Define Parameters
    nn_inputs = X_train.shape[1]
    nn_hidden = 50
    nn_targets = y_train.shape[1]
             
    n_epochs = 10000
    l_rate = 0.3
    momentum = 0.95
     
    # Train neural network for different epochs
    nn = NeuralNet(nn_inputs, nn_hidden, nn_targets, NNType.REGRESSION)
    nn.train(X_train, y_train, n_epochs, l_rate, momentum)
    results_train[i] = round(nn.evaluate(X_train, y_train),3)
    results_test[i] = round(nn.evaluate(X_test, y_test),3)
     
    print('Number of epochs: ' + str(n_epochs))
    print('\nAccuracy in training: ' + str(results_train[i]) + ' %')
    print('Accuracy in testing: ' + str(results_test[i]) + ' %\n')
 
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(test_set_size, results_train, 'r', label='Base de Treino')
plt.plot(test_set_size, results_test, 'b', label='Base de Teste')
plt.ylabel('RMSE ')
plt.xlabel('Tamanho da Base de Teste')
plt.title("RMSE para Diferentes Proporções de Treino e Teste")
plt.axis([test_set_size[0], test_set_size[-1], 0, 5])
plt.tight_layout()
ax.legend()
plt.grid()
plt.show()


# Get Data
X_train, X_test, y_train, y_test = NNData().get_music_data(.2)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())
 
# Define Parameters
nn_inputs = X_train.shape[1]
nn_hidden = 50
nn_targets = y_train.shape[1]
          
n_epochs = [50, 100, 500, 1000, 5000, 10000, 50000]
l_rate = 0.3
momentum = 0.95
  
# Train neural network for different epochs
nn = NeuralNet(nn_inputs, nn_hidden, nn_targets, NNType.REGRESSION)
results_train = np.zeros(len(n_epochs))
results_test = np.zeros(len(n_epochs))
for i in range(len(n_epochs)):
    nn.train(X_train, y_train, n_epochs[i], l_rate, momentum)
    results_train[i] = round(nn.evaluate(X_train, y_train),3)
    results_test[i] = round(nn.evaluate(X_test, y_test),3)
      
    print('Number of epochs: ' + str(n_epochs[i]))
    print('\nAccuracy in training: ' + str(results_train[i]) + ' %')
    print('Accuracy in testing: ' + str(results_test[i]) + ' %\n')
  
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(n_epochs, results_train, 'r', label='Base de Treino')
plt.plot(n_epochs, results_test, 'b', label='Base de Teste')
plt.ylabel('RMSE')
plt.xlabel('Número de Épocas')
plt.title("RMSE para Diferentes Quantidades de Épocas")
plt.axis([0, 10000, 0, 5])
plt.tight_layout()
labels = [str(x) for x in n_epochs]
ax.set_xticklabels(labels)
ax.legend()
plt.grid()
plt.show()


# Get Data
X_train, X_test, y_train, y_test = NNData().get_music_data(.2)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())
 
# Define Parameters
nn_inputs = X_train.shape[1]
nn_hidden = 50
nn_targets = y_train.shape[1]
          
n_epochs = 10000
l_rate = 0.3
momentum = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
  
# Train neural network for different epochs
nn = NeuralNet(nn_inputs, nn_hidden, nn_targets, NNType.REGRESSION)
results_train = np.zeros(len(momentum))
results_test = np.zeros(len(momentum))
for i in range(len(momentum)):
    nn.train(X_train, y_train, n_epochs, l_rate, momentum[i])
    results_train[i] = round(nn.evaluate(X_train, y_train),3)
    results_test[i] = round(nn.evaluate(X_test, y_test),3)
      
    print('Momentum: ' + str(momentum[i]))
    print('\nAccuracy in training: ' + str(results_train[i]) + ' %')
    print('Accuracy in testing: ' + str(results_test[i]) + ' %\n')
  
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(momentum, results_train, 'r', label='Base de Treino')
plt.plot(momentum, results_test, 'b', label='Base de Teste')
plt.ylabel('RMSE')
plt.xlabel('Momentum')
plt.title("RMSE para Diferentes Momentums")
plt.axis([.25, .85, 0, 5])
plt.tight_layout()
labels = [str(x) for x in momentum]
ax.set_xticklabels(labels)
ax.legend()
plt.grid()
plt.show()

# Get Data
X_train, X_test, y_train, y_test = NNData().get_music_data(.2)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())
  
# Define Parameters
nn_inputs = X_train.shape[1]
nn_hidden = 50
nn_targets = y_train.shape[1]
           
n_epochs = 10000
l_rate = [0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 1, 2, 3]
momentum = 0.95
   
# Train neural network for different epochs
nn = NeuralNet(nn_inputs, nn_hidden, nn_targets, NNType.REGRESSION)
results_train = np.zeros(len(l_rate))
results_test = np.zeros(len(l_rate))
for i in range(len(l_rate)):
    nn.train(X_train, y_train, n_epochs, l_rate[i], momentum)
    results_train[i] = round(nn.evaluate(X_train, y_train),3)
    results_test[i] = round(nn.evaluate(X_test, y_test),3)
       
    print('Learning Rate: ' + str(l_rate[i]))
    print('\nAccuracy in training: ' + str(results_train[i]) + ' %')
    print('Accuracy in testing: ' + str(results_test[i]) + ' %\n')
   
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(l_rate, results_train, 'r', label='Base de Treino')
plt.plot(l_rate, results_test, 'b', label='Base de Teste')
plt.ylabel('RMSE')
plt.xlabel('Taxa de Aprendizagem')
plt.title("RMSE para Diferentes Taxas de Aprendizagem")
plt.axis([.01, 3, 0, 5])
plt.tight_layout()
labels = [str(x) for x in l_rate]
ax.set_xticklabels(labels)
ax.legend()
plt.grid()
plt.show()