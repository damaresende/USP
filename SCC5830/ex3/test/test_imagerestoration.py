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
from math import ceil
from unittest.mock import patch
from matplotlib import pyplot as plt

from SCC5830.ex3.src import imagerestoration


class IMGRestorationTests(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        super(IMGRestorationTests, cls).setUpClass()
        
    def test_dispn_calculation(self):
        img = np.array([[171, 222, 130, 139, 239, 246,  64,  37, 114, 122,  25,  84, 152, 140,  37,   4, 176, 136],
                        [218, 88, 126, 134, 181, 211,  10,  13,  85,  38,   2,  91,  53, 151, 205,  63,  11,  60],
                        [162, 113, 206, 35, 136,  17,  30, 163,  23,   2, 178, 112, 220,  23, 232,   4,  61,  48],
                        [ 75, 161, 139, 211, 157,  84, 210, 47,  36,  23,  58, 220,  43, 130, 248,  48,  39, 152],
                        [ 14, 125, 145, 146, 155, 238, 226, 83, 159, 139, 188, 116, 178,  35, 231,  74,  29, 235],
                        [155,  69,  60,  20, 137, 234, 122, 103, 65,  65, 135,  88, 252,  32, 204,  71, 199,   4],
                        [208, 140, 120, 226, 222, 235, 148,  47, 12, 152, 223,  96,  40, 111, 202, 180, 159, 178],
                        [208, 185,  64, 137, 122, 195, 123, 205, 234, 132, 19,  56, 194, 102, 182, 104, 101, 180],
                        [ 72,  25, 156, 233, 154,  86, 243,  76,  70,  33, 211, 92,  38,  34,  66, 168, 131, 239],
                        [ 17, 169,  46,  53, 242, 132, 185, 209, 141,  54, 169, 64, 116,  50,  73, 173, 118, 237],
                        [136,  53, 210, 118,  60,  49, 173,  23,  14, 200,  30, 104, 249, 107, 237, 251, 130, 76],
                        [ 24, 165, 190, 181, 228, 210,  80,  31, 179, 175, 165, 236, 194,  65, 110,  98, 242, 58],
                        [202,  20, 213, 203, 212, 178, 137, 171,  70, 196, 245, 162,  88, 122, 207, 240, 151, 194],
                        [  1,  45, 252,  21,  20, 124,  19,  92, 210,  14,  53, 10, 185,  14, 113, 235,  26,  55],
                        [247,   1, 136, 192, 142, 146, 237,  23, 229, 114,  95, 110, 188,  55, 192, 226, 252,  81],
                        [190, 213,  83, 128, 240,  80, 155,  25,  49,  15, 116, 119, 148, 248, 179, 111, 203, 195],
                        [236,  61, 137,  69,  42,  84, 150,  46, 148, 110, 217, 249, 149, 238,   1, 187, 160,  26],
                        [230, 125,  61, 180, 135,  54,   1, 191,  64, 197, 217, 128,  88, 240,  65,  19, 106,  32]])
         
        user_input = ['0.15', '5', 'average']
        with patch('builtins.input', side_effect=user_input):
            filt = imagerestoration.DenoisingFilter()
            dispn = img[0:ceil(img.shape[0]/6 - 1), 0:ceil(img.shape[1]/6 -1)]
            dispn = filt.calc_disp(dispn)
            self.assertEqual(53.9508, round(dispn, 4))
             
        user_input = ['0.15', '5', 'robust']
        with patch('builtins.input', side_effect=user_input):
            filt = imagerestoration.DenoisingFilter()
            dispn = img[0:ceil(img.shape[0]/6 - 1), 0:ceil(img.shape[1]/6 -1)]
            dispn = filt.calc_disp(dispn)
            self.assertEqual(51, dispn)
    
    def test_denoising_polygons_average(self):
        user_input = ['files/polygons128.png', 'files/case2_45.png', '1', '0.15', '5', 'average']
        expected_output = 25.722
        left_bound = expected_output - expected_output * 0.2
        right_bound = expected_output + expected_output * 0.2
         
        with patch('builtins.input', side_effect=user_input):
            ref_img, deg_img, restored_img, rmse = imagerestoration.run_restoration()
            print('Test 1: Polygon denoising (average) RMSE: %s' % str(rmse))
             
            self.assertTrue(np.amax(restored_img) == np.max(deg_img))
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

    def test_denoising_polygons_robust(self):
        user_input = ['files/polygons128.png', 'files/case2_45.png', '1', '0.95', '5', 'robust']
        expected_output = 30.401
        left_bound = expected_output - expected_output * 0.2
        right_bound = expected_output + expected_output * 0.2
         
        with patch('builtins.input', side_effect=user_input):
            ref_img, deg_img, restored_img, rmse = imagerestoration.run_restoration()
            print('Test 2: Polygons denoising (robust) RMSE: %s' % str(rmse))
              
            self.assertTrue(np.amax(restored_img) == np.max(deg_img))
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
          
    def test_denoising_moon_robust(self):
        user_input = ['files/moon.jpg', 'files/case3_70.png', '1', '0.8', '5', 'robust']
        expected_output = 26.475
        left_bound = expected_output - expected_output * 0.2
        right_bound = expected_output + expected_output * 0.2
         
        with patch('builtins.input', side_effect=user_input):
            ref_img, deg_img, restored_img, rmse = imagerestoration.run_restoration()
            print('Test 3: Moon denoising (robust) RMSE: %s' % str(rmse))
              
            self.assertTrue(np.amax(restored_img) == np.max(deg_img))
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
         
    def test_denoising_moon_average(self):
        user_input = ['files/moon.jpg', 'files/case3_70.png', '1', '1.0', '5', 'average']
        expected_output = 26.640
        left_bound = expected_output - expected_output * 0.2
        right_bound = expected_output + expected_output * 0.2
          
        with patch('builtins.input', side_effect=user_input):
            ref_img, deg_img, restored_img, rmse = imagerestoration.run_restoration()
            print('Test 4: Moon denoising (average) RMSE: %s' % str(rmse))
              
            self.assertTrue(np.amax(restored_img) == np.max(deg_img))
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