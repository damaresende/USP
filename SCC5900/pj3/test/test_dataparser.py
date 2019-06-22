'''
Tests for module imgenerator

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 22, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Project of Algorithms Class (SCC5000)
'''
import unittest

from SCC5900.pj3.src.dataparser import DataParser


class DataParserUTests(unittest.TestCase):
    
    def test_get_labels(self):
        '''
        Tests if list of labels is correctly retrieved. In total the list must
        contain 12 labels.
        '''
        labels = DataParser.get_labels()
        
        self.assertEqual(12, len(labels))
        self.assertEqual('esquerda', labels[1])
        self.assertEqual('diagonal1baixo', labels[10])
    
    def test_training_set(self):
        '''
        Tests if the array of training data is correctly retrieved. The result must a be a
        tuple with a list of lists where the outer list has length 240 and the inner list has 
        length X, where X is the number of data points in the time series, and a list of labels
        of length 240 from values 1 to 12.
        '''
        train_X, train_Y = DataParser.get_training_set()
        
        self.assertEqual(240, len(train_X))
        self.assertEqual(240, len(train_Y))
        
        for value in train_X:
            isinstance(value, list)
            
        for value in train_Y:
            try:
                self.assertTrue(value >= 1 and value <= 12)
            except AssertionError:
                pass
     
    def test_test_set(self):
        '''
        Tests if the array of test data is correctly retrieved. The result must a be a
        tuple with a list of lists where the outer list has length 960 and the inner list has 
        length X, where X is the number of data points in the time series, and a list of labels
        of length 960 from values 1 to 12.
        '''
        test_X, test_Y = DataParser.get_test_set()
        
        self.assertEqual(960, len(test_X))
        self.assertEqual(960, len(test_Y))
        
        for value in test_X:
            isinstance(value, list)
            
        for value in test_Y:
            try:
                self.assertTrue(value >= 1 and value <= 12)
            except AssertionError:
                pass
        