'''
Created on Sep 30, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 3: MLP + RBF
@author: Damares Resende
'''

from math import sqrt
import numpy as np

class MLPNet():
    
    def __init__(self, nn_inputs, nn_hidden1, nn_hidden2, nn_targets):
        self.nn_inputs = nn_inputs
        self.nn_targets = nn_targets
            
        self.weights0 = np.random.uniform(-1, 1, (nn_inputs, nn_hidden1)) 
        self.weights1 = np.random.uniform(-1, 1, (nn_hidden1, nn_hidden2))
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
        
    def train(self, X, y, n_epochs, l_rate, momentum, verbose=False):
        for ep in range(n_epochs):
            self.feedforward(X)
            
            if verbose and ep % 1000 == 0:
                print('Error: ' + str(np.mean(np.abs(self.calc_rmse(y, self.y_hat)))))
            
            self.backpropagation(X, y, l_rate, momentum)
        
    def activation_fn(self, x):
        return 1 / (1 + np.exp(-x))
    
    def activation_fn_derivative(self, y_hat):
        return y_hat * (1 - y_hat)
    
    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.y_hat, axis=1)
    
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
        return self.calc_accuracy(y, y_hat)
    

class RBFNet():
    def __init__(self, nFeatures, nPrototypes, nClasses):
        self.spread = 0
        self.nClasses = nClasses
        self.nPrototypes = nPrototypes
        self.prototypes = np.zeros((0, nFeatures))
        self.weights = np.random.randn(self.nPrototypes * nClasses, nClasses)
    
    def clusterize(self, X, y):
        clusters = {}
        for idx, label in enumerate(y):
            label = np.array2string(label)
            if label not in clusters.keys():
                clusters[label] = [X[idx]]
            else:
                clusters[label].append(X[idx])
        return clusters
                
    def generateRandomPrototypes(self, X, y):
        clusters = self.clusterize(X, y)
        for key in clusters.keys():
            cluster = np.array(clusters[key])
            sample_idxs = np.random.randint(0,len(cluster), size = self.nPrototypes)
            self.prototypes = np.vstack([self.prototypes, cluster[sample_idxs,:]])
        return self.prototypes
    
    def compute_spread(self):
        dTemp = 0
        for i in range(0, self.nPrototypes * self.nClasses):
            for k in range(0, self.nPrototypes * self.nClasses):
                dist = np.square(np.linalg.norm(self.prototypes[i] - self.prototypes[k]))
                if dist > dTemp:
                    dTemp = dist
        self.spread = dTemp / np.sqrt(self.nPrototypes * self.nClasses)
        
    def activation_fn(self, input_, prototype):
        distance = np.square(np.linalg.norm(input_ - prototype))
        return np.exp(-(distance) / (np.square(self.spread)))
    
    def feedforward(self, input_):
        a_out = np.zeros(self.nPrototypes * self.nClasses)
        for ptt, prototype in enumerate(self.prototypes):
            neuronOut = self.activation_fn(input_, prototype)
            a_out[ptt] = neuronOut
        return a_out

    def train(self, X, y):
        self.generateRandomPrototypes(X, y)
        self.compute_spread()
        
        hiddenOut = np.zeros((X.shape[0], self.nPrototypes * self.nClasses))
        for idx, input_ in enumerate(X):
            hiddenOut[idx] = np.array(self.feedforward(input_))
    
        self.weights = np.dot(hiddenOut.T, y)
        
    def predict(self, X):
        y_hat = np.zeros((X.shape[0], self.nClasses))
        for idx, input_ in enumerate(X):
            max_idx = np.argmax(np.dot(np.array(self.feedforward(input_)).T, self.weights))
            y_hat[idx][max_idx] = 1
        return y_hat
    
    def calc_accuracy(self, y, y_hat):
        error = 0
        for i in range(y.shape[0]):
            if not (y[i] == y_hat[i]).all():
                error += 1
        return 100 - error * 100 / y.shape[0]
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        return self.calc_accuracy(y, y_hat)