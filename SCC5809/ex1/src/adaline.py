'''
Created on Aug 22, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 1: Neural network with only one perceptron.
@author: Damares Resende
'''

import numpy as np
from math import floor
import matplotlib.pyplot as plt

class AdalineTools(object):

    def print_all(self, dataset):
        """Prints all examples"""
        for i in range(len(dataset)):
            self.print_example(np.matrix(dataset[i][0]))
            print()

    def print_example(self, example):
        """Prints an example"""
        for i in range(example.shape[0]):
            for j in range(example.shape[1]):
                cell = example.item(i, j)
                if cell > 0:
                    print('# ', end='')
                else:
                    print('. ', end='')
            print()

    def build_input_dataset(self, inputs):
        """Converts the dataset into a bidimensional dataset where the rows are the
         examples and the columns the features.
         """
        n_examples = len(inputs)
        example_length = len(inputs[0][0][0]) * len(inputs[1][0][0])

        dataset = np.zeros((n_examples, example_length), dtype=int)
        for i in range(n_examples):
            dataset[i,:] = np.asarray(np.matrix(inputs[i][0])).reshape(-1)

        return dataset

    def build_labels_array(self, inputs):
        """Gets all labels"""
        n_examples = len(inputs)
        labels = np.zeros(n_examples, dtype=int)
        for i in range(n_examples):
            labels.itemset(i, inputs[i][1])

        return labels

    def split_training_test(self, dataset, lables, training_rate):
        """Returns randomly chosen datasets for training and testing"""
        n_examples = len(lables)
        np.random.shuffle(dataset)
        idx_limit = floor(n_examples*training_rate)

        training_set = dataset[:idx_limit][:]
        test_set = dataset[idx_limit:][:]
        training_lables = lables[:idx_limit]
        test_lables = lables[idx_limit:]

        return (training_set, training_lables, test_set, test_lables)

class Adaline(object):

    def __init__(self, eta = 0.001, epoch = 100):
        self.eta = eta # learning rate
        self.epoch = epoch

    def train(self, X, y):
        """Train model: applies the fitness function, adjusts weights and reduce errors"""
        np.random.seed(11)
        # creates a vector of integers of length n_columns + 1 in the range [-1, 1)
        self.weight_ = np.random.uniform(-1, 1, X.shape[1] + 1)
        self.error_ = []

        cost = 0
        for _ in range(self.epoch):

            output = self.activation_function(X)
            error = y - output

            self.weight_[0] += self.eta * sum(error)
            self.weight_[1:] += self.eta * X.T.dot(error)

            cost = 1./2 * sum((error**2))
            self.error_.append(cost)

        return self

    def test(self, X, y):
        """Predicts examples and compare the result with the original label"""
        if not hasattr(self, "weight_"):
            return

        error = 0
        result = self.predict(X)
        n_examples = len(y)

        for i in range(n_examples):
            if result[i] != y[i]:
                error += 1

        return (result, error*100./n_examples)

    def activation_function(self, X):
        """Calculate the output g(z)"""
        # multiplies each weight by an entrance, sums and add a bias
        return np.dot(X, self.weight_[1:]) + self.weight_[0]

    def predict(self, X):
        """Return the values -1 for and +1 for A upside down"""
        return np.where(self.activation_function(X) >= +1, +1, -1)

##########################################################################################################################################

inputs = [([[+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1], [-1, +1, +1, +1, -1], [-1, +1, -1, +1, -1], [-1, -1, +1, -1, -1]], +1), \
          ([[+1, -1, -1, -1, +1], [+1, -1, +1, -1, +1], [-1, +1, +1, +1, -1], [+1, +1, -1, +1, -1], [-1, -1, +1, -1, -1]], +1), \
          ([[+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1], [-1, -1, +1, +1, -1], [-1, +1, -1, +1, +1], [-1, -1, +1, -1, -1]], +1), \
          ([[+1, -1, -1, +1, +1], [+1, -1, -1, -1, -1], [-1, -1, +1, +1, -1], [-1, +1, -1, +1, -1], [-1, -1, +1, -1, -1]], +1), \
          ([[+1, -1, -1, -1, +1], [-1, -1, -1, -1, +1], [-1, +1, -1, +1, -1], [-1, -1, -1, +1, -1], [-1, -1, +1, -1, -1]], +1), \
          ([[+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1], [-1, +1, +1, +1, -1], [-1, +1, +1, +1, -1], [-1, -1, +1, -1, -1]], +1), \

          ([[-1, -1, +1, -1, -1], [-1, +1, -1, +1, -1], [-1, +1, +1, +1, -1], [+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1]], -1), \
          ([[-1, +1, +1, -1, -1], [-1, +1, -1, +1, +1], [-1, +1, +1, +1, -1], [+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1]], -1), \
          ([[-1, -1, +1, -1, -1], [-1, +1, -1, +1, -1], [+1, +1, +1, +1, -1], [+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1]], -1), \
          ([[-1, -1, +1, -1, -1], [-1, +1, -1, +1, -1], [-1, -1, +1, +1, -1], [+1, -1, -1, -1, +1], [-1, -1, -1, -1, +1]], -1), \
          ([[-1, -1, +1, -1, -1], [-1, +1, -1, +1, -1], [+1, +1, +1, +1, -1], [+1, -1, -1, +1, +1], [+1, -1, -1, -1, +1]], -1), \
          ([[-1, -1, -1, -1, -1], [-1, +1, -1, +1, +1], [-1, +1, +1, +1, -1], [+1, -1, -1, -1, +1], [+1, -1, -1, -1, +1]], -1)]

tools = AdalineTools()
tools.print_all(inputs)
X = tools.build_input_dataset(inputs)
y = tools.build_labels_array(inputs)

X_train, y_train, X_test, y_test = tools.split_training_test(X, y, 0.7)

classifier = Adaline(0.01, 20)
model = classifier.train(X_train, y_train)

classes, error_rate = classifier.test(X_test, y_test)
print('Prediction: ' + str(classes))
print('Labels: ' + str(y_test))
print('Error: ' + str(error_rate) + '%')

plt.plot(range(0, 20), model.error_)
plt.xlabel('Epochs')
plt.ylabel('Mean Square Error')
#plt.axis([0, 100, 0, 1])
plt.show()
