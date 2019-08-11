'''
Created on Sep 8, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Tests - Exercise 2: Neural networks with backpropagation - Enconding
@author: Damares Resende
'''

import unittest
import numpy as np

from SCC5809.ex2v2 import encoding as ecn
 
class EncodingTests(unittest.TestCase):
    
    def test_create_template(self):
        """
        Tests if an identity matrix is correctly created.
        """
        m_size = 10
        data = ecn.IDData(m_size)
        self.assertEqual(np.shape(data.ID_MATRIX), (m_size, m_size))
        self.assertEqual(sum(sum(data.ID_MATRIX)), m_size)
        self.assertTrue((sum(data.ID_MATRIX) == np.ones(m_size)).all())
        print('Finished testing template\n')
        
    def test_init_net_simple(self):
        """
        Tests the creation of a neural net with 2 inputs, 1 hidden layer 
        with 2 neurons, 1 output
        """
        net = ecn.NeuralNet(2, (2,), 1)
        self.assertEqual(2, len(net.weights.keys()))
        self.assertEqual((2, 3), np.shape(net.weights['h0']))
        self.assertEqual((1, 3), np.shape(net.weights['y']))
        print('Finished testing simple neural net init\n')
        
    def test_init_net_complex(self):
        """
        Tests the creation of a neural net with 10 inputs, 2 hidden layers, 
        one with 5 neurons, the other with 3 neurons, 2 output
        """
        net = ecn.NeuralNet(10, (5,3), 2)
        self.assertEqual(3, len(net.weights.keys()))
        self.assertEqual((5, 11), np.shape(net.weights['h0']))
        self.assertEqual((3, 6), np.shape(net.weights['h1']))
        self.assertEqual((2, 4), np.shape(net.weights['y']))
        print('Finished testing complex neural net init\n')
          
    def test_simple_net_forward(self):
        """
        Tests the feed forward process in a simple network. The weights
        are initalized to a fixed value in order to do that.
        """
        net = ecn.NeuralNet(2, (2,), 1)
        net.weights = self._set_initial_weights()
           
        dataset = [[1, 1]]
        targets = [[0]]
           
        net.train(dataset, targets, 0.5, 1)
        self.assertTrue(net.fit_values[0] == [0.3,  1.4])
        self.assertTrue(net.outputs[0] == [0.5744, 0.8022])
        self.assertTrue(net.fit_values[1] == [0.1922])
        self.assertTrue(net.outputs[1] == [0.5479])
        print('Finished testing simple neural net forward\n')
         
    def test_net_backpropagation_one_input(self):
        """
        Tests the backpropagation process in a simple network with one input.
        The weights are initalized to a fixed value in order to do that.
        """
        net = ecn.NeuralNet(2, (2,), 1)
        net.weights = self._set_initial_weights()
          
        dataset = [[1, 1]]
        targets = [[0]]
          
        net.train(dataset, targets, 0.5, 1)
        self.assertTrue((net.weights['h0'][0] == [-0.5934,  0.4066, 0.5066]).all())
        self.assertTrue((net.weights['h0'][1] == [-0.2097,  0.7903, 0.7903]).all())
        self.assertTrue((net.weights['y'][0] == [-0.3679,  -0.4390, 0.8456]).all())
        print('Finished testing backpropagation one input\n')
        
    def test_net_backpropagation_two_inputs(self):
        """
        Tests the backpropagation process in a simple network with two inputs.
        The weights are initalized to a fixed value in order to do that.
        """
        net = ecn.NeuralNet(2, (2,), 1)
        net.weights = self._set_initial_weights()
          
        dataset = [[1, 1], [0, 0]]
        targets = [[0], [0]]
          
        net.train(dataset, targets, 0.5, 1)
        self.assertTrue((net.weights['h0'][0] == [-0.5876,  0.4066, 0.5066]).all())
        self.assertTrue((net.weights['h0'][1] == [-0.2218,  0.7903, 0.7903]).all())
        self.assertTrue((net.weights['y'][0] == [-0.4256,  -0.4595, 0.8198]).all())
        print('Finished testing backpropagation two inputs\n')
    
    def test_net_backpropagation_four_inputs(self):
        """
        Tests the backpropagation process in a simple network with three inputs.
        The weights are initalized to a fixed value in order to do that.
        """
        net = ecn.NeuralNet(2, (2,), 1)
        net.weights = self._set_initial_weights()
          
        dataset = [[1, 1], [0, 0], [0, 1], [1, 0]]
        targets = [[0], [0], [1], [1]]
          
        net.train(dataset, targets, 0.5, 1)
        self.assertTrue((net.weights['h0'][0] == [-0.6018,  0.400,  0.499 ]).all())
        self.assertTrue((net.weights['h0'][1] == [-0.1969,  0.8027,  0.8028]).all())
        self.assertTrue((net.weights['y'][0] == [-0.2970, -0.3995,  0.9021]).all())
        print('Finished testing backpropagation four inputs\n')
        
    def _set_initial_weights(self):
        """
        Builds the initial weights structure for a neural net with 2 inputs, 1 hidden layer 
        with 2 neurons, 1 output
         
        Returns:
            @return: initial weights 
            @rtype: pandas data frame
        """
        weights = {'h0': None, 'y': None}
        weights['h0'] = np.array([[-0.6, 0.4, 0.5], [-0.2, 0.8, 0.8]])
        weights['y'] = np.array([[-0.3, -0.4, 0.9]])
         
        return weights