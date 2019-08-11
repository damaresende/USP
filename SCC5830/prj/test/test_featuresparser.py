'''
Tests for module featuresparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Feb 19, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import unittest
import numpy as np

from src.featuresparser import FeaturesParser


class AnnotationsParserTests(unittest.TestCase):
        
    def test_get_labels(self):
        '''
        Tests if array of labels is retrieved
        '''
        parser = FeaturesParser('../data/')
        labels = parser.get_labels()
         
        self.assertTrue(isinstance(labels, np.ndarray))
        self.assertEqual((3591,), labels.shape)
        for label in labels:
            self.assertTrue(label <= 50 and label >= 1)
             
    def test_get_labels_file_not_found(self):
        '''
        Tests if None is returned when file is not found
        '''
        parser = FeaturesParser('../data/dummy/')
        labels = parser.get_labels()
         
        self.assertEqual(None, labels)
         
    def test_get_viual_features(self):
        '''
        Tests if array of features is retrieved
        '''
        parser = FeaturesParser('../data/')
        features = parser.get_visual_features()
          
        self.assertEqual((3591, 2048), features.shape)
        self.assertTrue(sum(sum(features)) != 0)
        self.assertTrue(sum(sum(features)) != 1)
          
    def test_get_features_file_not_found(self):
        '''
        Tests if None is returned when file is not found
        '''
        parser = FeaturesParser('../data/dummy/')
        features = parser.get_visual_features()
          
        self.assertEqual(None, features)
          
    def test_get_semantic_features(self):
        '''
        Tests if semantic features are correctly retrieved
        '''
        parser = FeaturesParser('../data/')
        features = parser.get_semantic_features()
          
        self.assertEqual((3591, 24), features.shape)
        self.assertTrue(sum(sum(features)) > 0)
        
    def test_get_data(self):
        '''
        Tests if data dictionary is correctly retrieved
        '''
        parser = FeaturesParser('../data/')
        data = parser.get_data()
        
        self.assertEqual((2700, 2048), data['x_train_vis'].shape)
        self.assertEqual((2700, 24), data['x_train_sem'].shape)
        self.assertEqual((2700,), data['y_train'].shape)
        self.assertEqual((891, 2048), data['x_test_vis'].shape)
        self.assertEqual((891, 24), data['x_test_sem'].shape)
        self.assertEqual((891,), data['y_test'].shape)
