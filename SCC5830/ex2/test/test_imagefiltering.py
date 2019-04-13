'''
Tests for module imagefiltering

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 13, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import unittest
from unittest.mock import patch

from SCC5830.ex2.src import imagefiltering


class IMGFilteringTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super(IMGFilteringTests, cls).setUpClass()
        
    def test_1D_limiar_filter(self):
        '''
        Test limiarization
        '''
        user_input = ['files/baboon.png', '1', '200']
        expected_output = 93.3495
        
        with patch('builtins.input', side_effect=user_input):
            result = imagefiltering.run_filtering()
            self.assertEqual(expected_output, result)
            
    def test_1D_filter(self):
        '''
        Test 1D filtering
        '''
        user_input = ['files/arara.png', '2', '5', '-2 -1 0 1 2']
        expected_output = 68.2459
        
        with patch('builtins.input', side_effect=user_input):
            result = imagefiltering.run_filtering()
            self.assertEqual(expected_output, result)
            
    def test_2D_limiar_filter_3x3(self):
        '''
        Test 2D filtering with limiarization with 3x3 structuring element
        '''
        user_input = ['files/airplane.png', '3', '3', '-1 -1 -1', '-1 8 -1', '-1 -1 -1', '200']
        expected_output = 135.5598
        
        with patch('builtins.input', side_effect=user_input):
            result = imagefiltering.run_filtering()
            self.assertEqual(expected_output, result)
            
    def test_2D_limiar_filter_5x5(self):
        '''
        Test 2D filtering with limiarization with 5x5 structuring element
        '''
        user_input = ['files/flower.png', '3', '5', '-1 -1 -1 -1 -1', '-1 -1 -1 -1 -1', 
                      '-1 -1 24 -1 -1 ', '-1 -1 -1 -1 -1', '-1 -1 -1 -1 -1', '119']
        expected_output = 117.1761
        
        with patch('builtins.input', side_effect=user_input):
            result = imagefiltering.run_filtering()
            self.assertEqual(expected_output, result)
            
    def test_median_filter(self):
        '''
        Test median filter
        '''
        user_input = ['files/camera_saltpepper.png', '4', '5']
        expected_output = 43.1823
        
        with patch('builtins.input', side_effect=user_input):
            result = imagefiltering.run_filtering()
            self.assertEqual(expected_output, result)

if __name__ == '__main__':
    unittest.main()