'''
Tests for module imagerestoration

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 11, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import unittest
import numpy as np
from unittest.mock import patch
from matplotlib import pyplot as plt

from SCC5830.ex3.src import imagerestoration


class IMGRestorationTests(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        super(IMGRestorationTests, cls).setUpClass()
        
    def test_denoising_polygons_average(self):
        user_input = ['files/polygons128.png', 'files/case1_10.png' '1', '0.15', '5', '"average"']
        expected_output = 7.722
        left_bound = expected_output + expected_output * 0.2
        right_bound = expected_output + expected_output * 0.2
        
        with patch('builtins.input', side_effect=user_input):
            ref_img, deg_img, restored_img, rmse = imagerestoration.run_restoration()
            print('Test 1D Limiar RMSE: %s' % str(rmse))
            
            self.assertTrue(np.amax(restored_img) == 255)
            self.assertTrue(np.amin(restored_img) == 0)
            self.assertTrue(left_bound < rmse < right_bound)
            
        plt.figure()
        plt.subplot(131)
        plt.imshow(ref_img, cmap="gray", vmin=0, vmax=255)
        plt.subplot(132)
        plt.imshow(deg_img, cmap="gray", vmin=0, vmax=255)
        plt.subplot(133)
        plt.imshow(restored_img, cmap="gray", vmin=0, vmax=255)
        plt.show()

#     def test_denoising_polygons_robust(self):
#         user_input = ['files/polygons128.png', 'files/case2_45.png' '1', '0.95', '5', '"robust"']
#         expected_output = 8.401
#         
#     def test_denoising_moon_robust(self):
#         user_input = ['files/moon.jpg', 'files/case3_70.png' '1', '0.8', '5', '"robust"']
#         expected_output = 8.475
#         
#     def test_denoising_moon_average(self):
#         user_input = ['files/moon.jpg', 'files/case3_70.png' '1', '1.0', '5', '"average"']
#         expected_output = 8.640
#         
#     def test_deblurring_polygons_1(self):
#         user_input = ['files/polygons128.png', 'files/case5_5_1.png' '2', '0.00005', '5', '1.0']
#         expected_output = 9.960
#         
#     def test_deblurring_polygons_2(self):
#         user_input = ['files/polygons128.png', 'files/case6_3_1.png' '2', '0.02', '3', '1.0']
#         expected_output = 8.658
#         
#     def test_deblurring_moon_1(self):
#         user_input = ['files/moon.jpg', 'files/case7_7_125.png' '2', '0.00008', '7', '1.25']
#         expected_output = 9.562
#         
#     def test_deblurring_moon_2(self):
#         user_input = ['files/moon.jpg', 'files/case8_3_15.png' '2', '0.008', '3', '1.6']
#         expected_output = 9.154