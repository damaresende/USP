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
import numpy as np
from unittest.mock import patch
from matplotlib import pyplot as plt

from SCC5830.ex2.src import imagefiltering


class IMGFilteringTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super(IMGFilteringTests, cls).setUpClass()
         
#     def test_1D_limiar_filter(self):
#         '''
#         Test limiarization
#         '''
#         user_input = ['files/baboon.png', '1', '200']
#         expected_output = 93.3495
#            
#         with patch('builtins.input', side_effect=user_input):
#             img, filterd_img, rmse = imagefiltering.run_filtering()
#             self.assertTrue(np.amax(filterd_img) == 1)
#             self.assertTrue(np.amin(filterd_img) == 0)
#             self.assertEqual(expected_output, rmse)
#                
#         print('Test 1D Limiar RMSE: %s' % str(rmse))
#            
#         plt.figure()
#         plt.subplot(121)
#         plt.imshow(img, cmap="gray", vmin=0, vmax=255)
#         plt.subplot(122)
#         plt.imshow(filterd_img, cmap="gray", vmin=0, vmax=1)
#         plt.show()
              
    def test_1D_filter(self):
        '''
        Test 1D filtering
        '''
        user_input = ['files/arara.png', '2', '5', '-2 -1 0 1 2']
        expected_output = 68.2459
            
        with patch('builtins.input', side_effect=user_input):
            img, filterd_img, rmse = imagefiltering.run_filtering()
            self.assertTrue(expected_output - 10 < rmse < expected_output + 10)
                  
        print('Test 1D Filter RMSE: %s' % str(rmse))
            
        plt.figure()
        plt.subplot(121)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.subplot(122)
        plt.imshow(filterd_img, cmap="gray", vmin=0, vmax=255)
        plt.show()
          
    def test_circular_array(self):
        user_input = ['3', '5 3 2']
          
        example = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_result = np.array([9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1]) 
        with patch('builtins.input', side_effect=user_input):
            filt = imagefiltering.Filter1D()
            circ_array = filt.create_circular_array(example)
            self.assertTrue((expected_result == circ_array).all())
      
    def test_1D_filter_apply_filter(self):
        user_input = ['3', '5 3 2']
           
        example = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_result = np.array([[31, 23, 33], [43, 53, 63], [73, 83, 48]]) 
        with patch('builtins.input', side_effect=user_input):
            filt = imagefiltering.Filter1D()
            filtered_img = filt.apply_filter(example)
            self.assertTrue((expected_result == filtered_img).all())
          
#     def test_2D_limiar_filter_3x3(self):
#         '''
#         Test 2D filtering with limiarization with 3x3 structuring element
#         '''
#         user_input = ['files/airplane.png', '3', '3', '-1 -1 -1', '-1 8 -1', '-1 -1 -1', '200']
#         expected_output = 135.5598
#             
#         with patch('builtins.input', side_effect=user_input):
#             img, filterd_img, rmse = imagefiltering.run_filtering()
#             self.assertTrue(expected_output - 10 < rmse < expected_output + 10)
#                 
#         print('Test 2D Limiar Filter 3x3 RMSE: %s' % str(rmse))
           
#         plt.figure()
#         plt.subplot(121)
#         plt.imshow(img, cmap="gray", vmin=0, vmax=255)
#         plt.subplot(122)
#         plt.imshow(filterd_img, cmap="gray", vmin=0, vmax=1)
#         plt.show()
                
#     def test_2D_limiar_filter_5x5(self):
#         '''
#         Test 2D filtering with limiarization with 5x5 structuring element
#         '''
#         user_input = ['files/flower.png', '3', '5', '-1 -1 -1 -1 -1', '-1 -1 -1 -1 -1', 
#                       '-1 -1 24 -1 -1 ', '-1 -1 -1 -1 -1', '-1 -1 -1 -1 -1', '119']
#         expected_output = 117.1761
#             
#         with patch('builtins.input', side_effect=user_input):
#             img, filterd_img, rmse = imagefiltering.run_filtering()
#             self.assertTrue(expected_output - 10 < rmse < expected_output + 10)
#                  
#         print('Test 2D Limiar Filter 5x5 RMSE: %s' % str(rmse))
#             
#         plt.figure()
#         plt.subplot(121)
#         plt.imshow(img, cmap="gray", vmin=0, vmax=255)
#         plt.subplot(122)
#         plt.imshow(filterd_img, cmap="gray", vmin=0, vmax=1)
#         plt.show()
              
    def test_median_filter(self):
        '''
        Test median filter
        '''
        user_input = ['files/camera_saltpepper.png', '4', '5']
        expected_output = 43.1823
           
        with patch('builtins.input', side_effect=user_input):
            img, filterd_img, rmse = imagefiltering.run_filtering()
            self.assertTrue(expected_output - 10 < rmse < expected_output + 10)
                
        print('Test Median Filter RMSE: %s' % str(rmse))

        plt.figure()
        plt.subplot(121)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.subplot(122)
        plt.imshow(filterd_img, cmap="gray", vmin=0, vmax=255)
        plt.show()
    
    def test_simple_filter_small_image(self):
        '''
        Tests simple filter with 5x5 image
        '''
        user_input = ['3', '-1 -1 -1', '-1 8 -1', '-1 -1 -1', '200']
        img = np.array([[ 23,   5,  39,  45,  50], 
                        [ 70,  88,  12, 100, 110],
                        [130, 145, 159, 136, 137],
                        [ 19, 200, 201, 220, 203],
                        [ 25,  26,  27,  28, 209],
                        [131,  32, 133,  34, 135]])
         
        expected_img = np.array([[  23,    5,   39,   45,   50],
                                [  70,  121, -621,  112,  110],
                                [ 130,  281,  170,  -54,  137],
                                [  19,  868,  667,  660,  203],
                                [  25, -560, -658, -938,  209],
                                [ 131,   32,  133,   34,  135]])
         
        with patch('builtins.input', side_effect=user_input):
            filt = imagefiltering.FilterFactory.init_filter(3)
            filterd_img = filt.simple_filter(img)
            self.assertTrue((expected_img == filterd_img).all())
    
    def test_median_filter_small_image(self):
        '''
        Tests median filter with 5x5 image
        '''
        user_input = ['3']
        img = np.array([[ 23,   5,  39,  45,  50], 
                        [ 70,  88,  12, 100, 110],
                        [130, 145, 159, 136, 137],
                        [ 19, 200, 201, 220, 203],
                        [ 25,  26,  27,  28, 209],
                        [131,  32, 133,  34, 135]])
        
        expected_img = np.array([[  0,  12,  12,  39,   0],
                                [ 23,  70,  88, 100,  50],
                                [ 70, 130, 145, 137, 110],
                                [ 25, 130, 145, 159, 136],
                                [ 25,  32,  34, 135,  34],
                                [  0,  26,  27,  28,   0]])
        
        with patch('builtins.input', side_effect=user_input):
            filt = imagefiltering.FilterFactory.init_filter(4)
            filterd_img = filt.apply_filter(img)
            self.assertTrue((expected_img == filterd_img).all())
        
        
if __name__ == '__main__':
    unittest.main()