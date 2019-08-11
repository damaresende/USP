'''
Created on Sep 11, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 2: Neural networks with backpropagation - Enconding Run
@author: Damares Resende
'''

from math import log
from SCC5809.ex2v2 import encoding as ecn

import matplotlib.pyplot as plt

N = 10
n_epochs = 50
learning_rate = 0.5
id_matrix = ecn.IDData(N).ID_MATRIX # creates a template with an identity 10x10 matrix

error = 1
model = None
plt.subplot(1, 2, 1)
for test_idx in range(10):
    test_dataset = id_matrix[test_idx]
    train_dataset = [inst for idx, inst in enumerate(id_matrix) if idx != test_idx]

    net = ecn.NeuralNet(N, (round(log(N, 2)),), N)
    net.train(train_dataset, train_dataset, learning_rate, n_epochs)
    
    if net.errors[-1] < error:
        model = net
        error = net.errors[-1] 
    

plt.plot(range(0, n_epochs), model.errors)

plt.xlabel('Epochs')
plt.ylabel('Root Mean Square Error')
plt.title("Root Mean Squared Error vs. Epochs")
plt.tight_layout()
plt.grid()
plt.axis([0, n_epochs, 0, 1])


plt.subplot(1, 2, 2)
outputs, errors = model.test(id_matrix, id_matrix)
acc = [1 - err for err in errors]
plt.plot(range(0, N), acc)

plt.xlabel('Instance')
plt.ylabel('Accuracy')
plt.title("Accuracy Per Instance")
plt.tight_layout()
plt.grid()
plt.axis([0, N, 0, 1])

plt.show()

