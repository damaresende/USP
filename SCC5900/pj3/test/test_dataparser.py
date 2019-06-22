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
        Tests if the array of training data is correctly retrieved. The result must
        be a 2D array with 240 and 32 columns. The first column must contain values 
        ranging from 1 to 12.
        '''
        train_data = DataParser.get_training_set()
        
        self.assertEqual((240, 32), train_data.shape)
        for value in train_data[:,0]:
            self.assertTrue(1 >= value <= 12)
    
    def test_test_set(self):
        '''
        Tests if the array of test data is correctly retrieved. The result must be
        a 2D array with 960 and 32 columns. The first column must contain values 
        ranging from 1 to 12.
        '''
        test_data = DataParser.get_test_set()
        
        self.assertEqual((240, 32), test_data.shape)
        for value in test_data[:,0]:
            self.assertTrue(1 >= value <= 12)
        