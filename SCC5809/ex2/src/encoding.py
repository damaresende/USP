'''
Created on Sep 8, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 2: Neural networks with backpropagation - Enconding
@author: Damares Resende
'''

import copy
import numpy as np
from math import sqrt, exp

class IDData():
    
    def __init__(self, m_size):
        self.matrix_size = m_size
        self._create_template()
        
    def _create_template(self):
        """
        Creates an identity matrix of size matrix_size.
        """
        self.ID_MATRIX = np.zeros((self.matrix_size,self.matrix_size))
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if i == j:
                    self.ID_MATRIX[i][j] = 1
                    
class NeuralNet():
    
    def __init__(self, n_inputs, hidden_layers, n_outputs):
        """
        Builds the neural net and initializes all weights to zero. For each
        neuron in a hidden layer and in the output layer, there is a bias as an
        extra input. The network is fully connected.

        Args:
            @param n_inputs: number of inputs
            @type n_inputs: integer
            @param hidden_layers: number of neurons per hidden layer 
            @type hidden_layers: tuple
            @param n_outputs: number of outputs 
            @type n_outputs: integer
        """
        self.hidden_layers = hidden_layers
        self.weights = {}
        self.weights['h0'] = np.zeros((hidden_layers[0], n_inputs + 1))
        
        for i in range(1, len(hidden_layers)):
            self.weights['h' + str(i)] = np.zeros((hidden_layers[i], hidden_layers[i-1] + 1))
    
        self.weights['y'] = np.zeros((n_outputs, hidden_layers[-1] + 1))
        
    def train(self, inputs, targets, l_rate, n_epochs):
        """
        Trains the network.
        TODO: generalize to accecpt more than one hidden layer
        TODO: add the multiple ephocs option

        Args:
            @param inputs: input features with instances in rows and features in columns
            @type inputs: matrix
            @param targets: expected outputs 
            @type targets: list
            @param l_rate: learning rate 
            @type l_rate: int, float
        """
        self.errors = []
        self.l_rate = l_rate
        
        for ep in range(n_epochs):
            total_loss = 0
            print('\nEpoch: ' + str(ep))
            
            for i in range(len(inputs)):
                self.fit_values = []
                self.outputs = []
            
                self._feed_forward(inputs[i])
                
                print('Instance: ' + str(inputs[i]))
                print('Output: ' + str(self.outputs[-1]))
                
                self._back_propagation(inputs[i], targets[i])
                
                total_loss += self._calc_loss(targets[i])  
            self.errors.append(total_loss / len(inputs))

    def _calc_loss(self, targets):
        """
        Computes the Root Mean Square Error in respect to the neural net outputs

        Args:
            @param targets: expected outputs
            @type targets: 1D array
        Returns:
            @return: RMSE
            @rtype: float
        """
        error = 0
        for idx in range(len(targets)):
            error += (targets[idx] - self.outputs[-1][idx]) * (targets[idx] - self.outputs[-1][idx])
        return sqrt(error / len(targets)) 
    
    def _feed_forward(self, input_):
        """
        Computes the neurons values based on input weights and the correspondent
        output based on the activation function

        Args:
            @param input_: input
            @type input_: 1D array
        """
        for layer in self.weights.keys():
            values = []
            outputs = []
            n_layer_neurons = self.weights[layer].shape[0]
            
            for neuron in range(n_layer_neurons):
                fit_value = self._fit(input_, layer, neuron)
                values.append(fit_value)
                outputs.append(self._activation_fn(fit_value))
            
            self.fit_values.append(values)
            self.outputs.append(outputs)
            input_ = outputs
    
    def _back_propagation(self, input_, target):
        """
        Updates the weights of the neural net

        Args:
            @param input_: input
            @type input_: 1D array
            @param target: expected output
            @type target: 1D array
        """
        
        # input with bias
        b_input_ = np.ones(len(input_) + 1)
        b_input_[1:] = input_
        
        # output with bias
        b_output_ = np.ones(len(self.outputs[-2]) + 1)
        b_output_[1:] = self.outputs[-2]
        
        y_layer = 'y'
        n_outputs = self.weights[y_layer].shape[0]
        delta = np.zeros(n_outputs)
        
        # backpropagation - output layer
        bck_weights = copy.deepcopy(self.weights) 
        for neuron in range(n_outputs):
            delta[neuron] = (target[neuron] - self.outputs[-1][neuron]) * self.outputs[-1][neuron] * (1 - self.outputs[-1][neuron])

            for n in range(self.weights[y_layer][0].shape[0]):
                self.weights[y_layer][neuron][n] = round(bck_weights[y_layer][neuron][n] + self.l_rate * delta[neuron] * b_output_[n], 4)                  

        # backpropagation - hidden layer
        shift = 1
        for layer in list(self.weights.keys())[:-1]:
            n_layer_neurons, n_layer_weights = self.weights[layer].shape
            
            for h_neuron in range(n_layer_neurons):
                error = 0
                for y_neuron in range(n_outputs):
                    error += delta[y_neuron] * bck_weights[y_layer][y_neuron][h_neuron + 1]
                
                for h_weight in range(n_layer_weights):
                    self.weights[layer][h_neuron][h_weight] = round(bck_weights[layer][h_neuron][h_weight] \
                                                    + self.l_rate * error * self.outputs[-1 - shift][h_neuron] \
                                                     * (1 - self.outputs[-1 - shift][h_neuron]) * b_input_[h_weight], 4)
            shift += 1

    def _fit(self, input_, layer_id, neuron_id):
        """
        Computes the value of the neuron based on the weights

        Args:
            @param input_: inputs for the specified neuron in the specified layer
            @type input_: list
            @param layer_id: layer id 
            @type layer_id: integer
            @param neuron_id: neuron id 
            @type neuron_id: integer
        Returns:
            @return: calculated value
            @rtype: float
        """
        value = self.weights[layer_id][neuron_id][0] + \
            np.dot(self.weights[layer_id][neuron_id][1:], input_)
        return round(value, 4)
    
    def _activation_fn(self, value):
        """
        Computes the output value based on the sigmoid function

        Args:
            @param value: neuron value
            @type value: int, float
        Returns:
            @return: calculated value
            @rtype: float
        """
        return round(1 / (1 + exp(-value)),4)
    
    def test(self, inputs, targets):
        """
        Tests the neural net based on the training weights

        Args:
            @param inputs: inputs
            @type inputs: 2D array
        """
        outputs = []
        errors = []
        
        for idx in range(len(inputs)):
            self._feed_forward(inputs[idx])
            outputs.append(self.outputs[-1])
            errors.append(self._calc_loss(targets[idx]))
            
        return outputs, errors