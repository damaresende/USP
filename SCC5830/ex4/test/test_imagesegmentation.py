'''
Tests for module imagesegmentation

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 10, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import unittest
import numpy as np
from math import ceil
from unittest.mock import patch
from matplotlib import pyplot as plt

from SCC5830.ex4.src import imagesegmentation


class IMGSegmentationTests(unittest.TestCase):   
    @classmethod
    def setUpClass(cls):
        super(IMGSegmentationTests, cls).setUpClass()
     
    def test_segmentation(self):
        user_input = ['files/test_image.png', 'files/ref_image.jpg', '4', '6', '16', '25']
        expected_output = 2.8527
        left_bound = expected_output - expected_output * 0.2
        right_bound = expected_output + expected_output * 0.2
        
        with patch('builtins.input', side_effect=user_input):
            ipt_img, ref_img, cmp_img, rmse = imagesegmentation.run_segmentation()
            print('RMSE: %s' % str(rmse))
              
#             self.assertTrue(np.amax(cmp_img) == np.max(ipt_img))
#             self.assertTrue(np.amin(cmp_img) == 0)
#             self.assertTrue(left_bound < rmse < right_bound)
        
        plt.figure()
        plt.subplot(131)
        plt.title('Input')
        plt.imshow(ipt_img.astype(np.uint8))
        plt.subplot(132)
        plt.title('Reference')
        plt.imshow(ref_img.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        plt.subplot(133)
        plt.title('Output')
        plt.imshow(cmp_img.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        plt.show()