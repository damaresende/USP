'''
Created on Sep 30, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 3: MLP + RBF
@author: Damares Resende
'''

import numpy as np
from NNDataset import NNData
from NNModels import RBFNet, MLPNet
from matplotlib import pyplot as plt

def main():
    nn_turns = 10
    log = {'RBF TN': np.zeros(nn_turns), 'RBF TS': np.zeros(nn_turns), \
           'MLP TN': np.zeros(nn_turns), 'MLP TS': np.zeros(nn_turns)}
    
    for i in range(nn_turns):
        print('############## Running turn ' + str(i + 1) + ' ##############\n')
        
        # Getting the data
        X, y = NNData.getSeedsData(normalize = True, binarize = True)
        X_train, X_test, y_train, y_test = NNData.stratifiedSplit(X, y, 0.2)
        
        # RBF Neural Net
        rbf = RBFNet(X_train.shape[1], 20, y_train.shape[1])
        rbf.train(X_train, y_train)
        
        log['RBF TN'][i] = rbf.evaluate(X_train, y_train)
        log['RBF TS'][i] = rbf.evaluate(X_test, y_test)
        
        print('>> RBF Accuracy - training: ' + str(round(log['RBF TN'][i],3)) + ' %')
        print('>> RBF Accuracy - testing: ' + str(round(log['RBF TS'][i],3)) + ' %\n')
        
        rbf = RBFNet(X_train.shape[1], 1, y_train.shape[1])
        rbf.train(X_train, y_train)
        
        # MLP Neural Net
        mlp = MLPNet(nn_inputs=X_train.shape[1], nn_hidden1=15, nn_hidden2=5, nn_targets=y_train.shape[1])
        mlp.train(X_train, y_train, n_epochs=20000, l_rate=0.3, momentum=0.75)
        
        log['MLP TN'][i] = mlp.evaluate(X_train, y_train)
        log['MLP TS'][i] = mlp.evaluate(X_test, y_test)
        
        print('>> MLP Accuracy - training: ' + str(round(log['MLP TN'][i],3)) + ' %')
        print('>> MLP Accuracy - testing: ' + str(round(log['MLP TS'][i],3)) + ' %\n')
        
        
    plt.plot(range(nn_turns), log['RBF TN'], '-o', label='RBF Treinamento')
    plt.plot(range(nn_turns), log['RBF TS'], '-o', label='RBF Teste')
    plt.plot(range(nn_turns), log['MLP TN'], '-o', label='MLP Treinamento')
    plt.plot(range(nn_turns), log['MLP TS'], '-o', label='MLP Teste')
    plt.legend()
    
    plt.xlabel('Rodadas')
    plt.ylabel('Acurácia (%)')
    plt.title("Performace da MLP vs RBF")
    plt.tight_layout()
    plt.show()
    
    print('############## The End. Bye =) ##############')

if __name__ == '__main__':
    main()