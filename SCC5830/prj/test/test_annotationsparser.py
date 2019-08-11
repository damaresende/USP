'''
Tests for module annotationsparser

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 28, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import unittest
import pandas as pd

from src.annotationsparser import AnnotationsParser


class AnnotationsParserTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        '''
        Initializes Annotations Parser
        '''
        cls.parser = AnnotationsParser('../data/')
        
    def test_get_labels(self):
        '''
        Tests if list of labels is retrieved
        '''
        labels = self.parser.get_labels()
        
        self.assertEqual(12, len(labels))
        self.assertEqual('dalmatian', labels.get(5))
        self.assertEqual('spider+monkey', labels.get(17))
        self.assertEqual('lion', labels.get(43))
        
    def test_get_labels_file_not_found(self):
        '''
        Tests if empty list is returned when file is not found
        '''
        parser = AnnotationsParser('../data/dummy/')
        labels = parser.get_labels()
        
        self.assertEqual(0, len(labels))
        
    def test_get_predicates(self):
        '''
        Tests if list of attributes is retrieved
        '''
        predicates = self.parser.get_predicates()
        
        self.assertEqual(85, len(predicates))
        self.assertEqual('orange', predicates[5])
        self.assertEqual('tree', predicates[76])
        self.assertEqual('hands', predicates[19])
        
    def test_get_predicates_file_not_found(self):
        '''
        Tests if empty list is returned when file is not found
        '''
        parser = AnnotationsParser('../data/dummy/')
        predicates = parser.get_predicates()
        
        self.assertEqual(0, len(predicates))    
    
    def test_get_attributes(self):
        '''
        Tests if data frame with attributes is retrieved
        '''
        attributes = self.parser.get_attributes()
        
        self.assertTrue(isinstance(attributes, pd.DataFrame))
        self.assertEqual(list(self.parser.get_labels().keys()), list(attributes.index.values))
        self.assertEqual((12,), attributes['toughskin'].values.shape)
        self.assertEqual((24,), attributes.loc[20].values.shape)
        
    def test_attributes_content(self):
        '''
        Tests if data frame with attributes have reasonable values
        '''
        attributes = self.parser.get_attributes()
        
        for label in self.parser.get_labels():
            self.assertTrue(sum(attributes.loc[label].values) < 2400)
            for value in attributes.loc[label].values:
                self.assertTrue(value >= -1 and value <= 100)
