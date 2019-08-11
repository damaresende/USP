'''
Created on Nov 18, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project III: Adaptative PCA
@author: Damares Resende
'''

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_data(x, y, flag):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Componente Principal 1', fontsize = 15)
    ax.set_ylabel('Componente Principal 2', fontsize = 15)
    ax.set_title('PCA com 2 componentes', fontsize = 20)
    targets = [0,1,2]
    colors = ['r', 'g', 'b']
    
    for target, color in zip(targets,colors):
        indicesToKeep = y == target
        ax.scatter(x[:,0][indicesToKeep[:,0]]
                   , x[:,1][indicesToKeep[:,0]]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    
    fig_name = 'pca_plot_%s.png' % flag
    fig.savefig(os.path.join(os.getcwd(), fig_name), dpi=fig.dpi)
    print('PCA plot saved to %s\n' % os.path.join(os.getcwd(), fig_name)) 


def plot_results(n_epochs, history, flag):
    fig = plt.figure()
    plt.plot([v for v in range(n_epochs)], np.array(history.history['loss']))
    plt.plot([v for v in range(n_epochs)], np.array(history.history['acc']))
    plt.ylabel('Loss / Accuracy (%)')
    plt.xlabel('Epochs')
    plt.legend(['loss', 'accuracy'], loc='upper left')
    
    fig_name = 'results_%s.png' % flag
    fig.savefig(os.path.join(os.getcwd(), fig_name), dpi=fig.dpi)
    print('Results saved to %s\n' % os.path.join(os.getcwd(), fig_name))


def run_wine_mlp(x_train, y_train, x_test, y_test, flag, n_epochs, plot=False):
    model = keras.Sequential([
        keras.layers.Dense(60, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=n_epochs)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    if plot:
        plot_results(n_epochs, history, flag)
    return test_loss, test_acc
    

def calc_pca(data, n_comps=2, standardize=False):
    if standardize:
        data = StandardScaler().fit_transform(data)
        
    data = data.mean(axis=0) - data
    cov_mat = np.cov(data, rowvar=False)
    evals, evecs = np.linalg.eig(cov_mat)
    
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    evecs = evecs[:, :n_comps]
    
    return np.dot(data, evecs), evals, evecs


def adaptative_pca(x, l_rate, n_epochs, n_comps, n_cycles, standardize):
    if standardize:
        x = StandardScaler().fit_transform(x)

    W = np.random.uniform(-0.01, 0.01, size=(x.shape[1], n_comps))
    V = np.triu(np.random.uniform(-0.01, 0.01, size=(n_comps, n_comps)))
    np.fill_diagonal(V, 0.0)
    
    # Training process
    for _ in range(n_epochs):
        for i in range(x.shape[0]):
            y_p = np.zeros((n_comps, 1))
            xi = np.expand_dims(x[i], 1)
    
            for _ in range(n_cycles):
                y = np.dot(W.T, xi) + np.dot(V, y_p)
                y_p = y.copy()
                
            dW = np.zeros((x.shape[1], n_comps))
            dV = np.zeros((n_comps, n_comps))
            
            for t in range(n_comps):
                y2 = np.power(y[t], 2)
                dW[:, t] = np.squeeze((y[t] * xi) + (y2 * np.expand_dims(W[:, t], 1)))
                dV[t, :] = -np.squeeze((y[t] * y) + (y2 * np.expand_dims(V[t, :], 1)))
    
            W += (l_rate * dW)
            V += (l_rate * dV)
            
            V = np.tril(V)
            np.fill_diagonal(V, 0.0)
            
            W /= np.linalg.norm(W, axis=0).reshape((1, n_comps))
        
    # Compute all output components
    Y_comp = np.zeros((x.shape[0], n_comps))
    for i in range(x.shape[0]):
        y_p = np.zeros((n_comps,1))
        xi = np.expand_dims(x[i], 1)
    
        for _ in range(n_cycles):
            Y_comp[i] = np.squeeze(np.dot(W.T, xi) + np.dot(V.T, y_p))
            y_p = y.copy()
            
    return Y_comp