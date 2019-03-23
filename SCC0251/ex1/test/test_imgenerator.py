'''
Tests for module imgenerator

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 17, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC0251)
'''

import random
import unittest
import numpy as np
from math import sin, cos, pow

from SCC0251.ex1.src.imgenerator import IMGenerator


class IMGeneratorTests(unittest.TestCase):
    
    def test_func_one(self):
        '''
        Test function 1
        '''
        C = 1024
        img = IMGenerator('ex1.npy', C, 1, 1, 720, 6, 1)()
         
        self.assertEqual((C, C), img.shape)
        for _ in range(0, 100):
            x = random.randint(0, C - 1)
            y = random.randint(0, C - 1)
            self.assertEqual(x * y + 2 * y, img[x][y])
             
    def test_func_two(self):
        '''
        Test function 2
        '''
        Q = 32
        C = 720
        img = IMGenerator('ex2.npy', C, 2, Q, 360, 4, 1)()
         
        self.assertEqual((C, C), img.shape)
        for _ in range(0, 100):
            x = random.randint(0, C - 1)
            y = random.randint(0, C - 1)
            self.assertAlmostEqual(abs(cos(x / Q) + 2 * sin(y/Q)), img[x][y], places=5)
             
    def test_func_three(self):
        '''
        Test function 3
        '''
        C = 720
        Q = 1001
        img = IMGenerator('ex3.npy', 720, 3, 1001, 256, 3, 1)()
         
        self.assertEqual((C, C), img.shape)
        for _ in range(0, 100):
            x = random.randint(0, C - 1)
            y = random.randint(0, C - 1)
            self.assertAlmostEqual(abs(3 * (x / Q) - pow(y/Q, 1/3)), img[x][y], places=5)
             
    def test_func_four(self):
        '''
        Test function 4
        '''
        S = 13
        C = 1024
        img = IMGenerator('ex4.npy', 1024, 4, 1, 256, 5, S)()
        self.assertEqual((C, C), img.shape)
             
    def test_func_five(self):
        '''
        Test function 5
        '''
        C = 500
        S = 6666
        random.seed(S)
        img = IMGenerator('ex5.npy', C, 5, 1, 250, 8, S)()
        self.assertEqual((C, C), img.shape)
        
    def test_downsamping(self):
        '''
        Test image downsampling
        '''
        img = np.array([[5, 15, 36, 0], [18, 0, 0, 1], [0, 100, 154, 0], [0, 99, 159, 100]], dtype=np.uint8)
        ref = np.array([[5, 36], [0, 154]], dtype=np.uint8)

        gen = IMGenerator('', 4, 1, 1, 2, 1, 1)
        self.assertTrue((ref == np.uint8(gen.downsampling(img))).all())
        
    def test_normalize(self):
        """
        Test image normalization
        """
        gen = IMGenerator('', 1024, 1, 1, 2, 1, 1)
        
        self.assertTrue(np.amax(gen.normalize(gen(), 8)) < pow(2, 8))
        self.assertTrue(np.amin(gen.normalize(gen(), 8)) >= 0)
