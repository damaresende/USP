'''
Tests for module imgenerator

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 17, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC0251)
'''

import os
import imageio
import unittest
import numpy as np


from SCC0251.ex1.src.imgenerator import IMGenerator


class IMGeneratorTests(unittest.TestCase):
    
    def test_func_one(self):
        '''
        Test function 1
        '''
        gen = IMGenerator('ex1.npy', 1024, 1, 1, 720, 6, 1)
        img = gen.generate_img()
        
        ref_np = (np.load(os.path.join(os.getcwd(), 'files/ex1.npy'))).astype(np.uint8)
        ref_img = imageio.imread(os.path.join(os.getcwd(), 'files/ex1_xy.png'))
        ref_img = ref_img.astype(np.uint8)
          
        print('Function 1 PNG: %s' % gen.calc_rmse(img, ref_img))
        self.assertTrue(gen.calc_rmse(img, ref_img) < 1000)
        
        print('Function 1 NPY: %s' % gen.calc_rmse(img, ref_np))
        self.assertTrue(gen.calc_rmse(img, ref_np) < 1000)
              
    def test_func_two(self):
        '''
        Test function 2
        '''
        gen = IMGenerator('ex2.npy', 720, 2, 32, 360, 4, 1)
        img = gen.generate_img()
        
        ref_np = (np.load(os.path.join(os.getcwd(), 'files/ex2.npy'))).astype(np.uint8)
        ref_img = imageio.imread(os.path.join(os.getcwd(), 'files/ex2_sin.png'))
        ref_img = ref_img.astype(np.uint8)
          
        print('Function 2 PNG: %s' % gen.calc_rmse(img, ref_img))
        self.assertTrue(gen.calc_rmse(img, ref_img) < 1000)
        
        print('Function 2 NPY: %s' % gen.calc_rmse(img, ref_np))
        self.assertTrue(gen.calc_rmse(img, ref_np) < 1000)
 
    def test_func_three(self):
        '''
        Test function 3
        '''
        gen = IMGenerator('ex3.npy', 720, 3, 1001, 256, 3, 1)
        img = gen.generate_img()
        
        ref_np = (np.load(os.path.join(os.getcwd(), 'files/ex3.npy'))).astype(np.uint8)
        ref_img = imageio.imread(os.path.join(os.getcwd(), 'files/ex3_quad.png'))
        ref_img = ref_img.astype(np.uint8)
          
        print('Function 3 PNG: %s' % gen.calc_rmse(img, ref_img))
        self.assertTrue(gen.calc_rmse(img, ref_img) < 1000)
        
        print('Function 3 NPY: %s' % gen.calc_rmse(img, ref_np))
        self.assertTrue(gen.calc_rmse(img, ref_np) < 1000)
                 
    def test_func_four(self):
        '''
        Test function 4
        '''
        gen = IMGenerator('ex4.npy', 1024, 4, 1, 256, 5, 13)
        img = gen.generate_img()
        
        ref_np = (np.load(os.path.join(os.getcwd(), 'files/ex4.npy'))).astype(np.uint8)
        ref_img = imageio.imread(os.path.join(os.getcwd(), 'files/ex4_rand.png'))
        ref_img = ref_img.astype(np.uint8)
          
        print('Function 4 PNG: %s' % gen.calc_rmse(img, ref_img))
        self.assertTrue(gen.calc_rmse(img, ref_img) < 1000)
        
        print('Function 4 NPY: %s' % gen.calc_rmse(img, ref_np))
        self.assertTrue(gen.calc_rmse(img, ref_np) < 1000)
           
    def test_func_five(self):
        '''
        Test function 5
        '''
        gen = IMGenerator('ex5.npy', 500, 5, 1, 250, 8, 6666)
        img = gen.generate_img()
        
        ref_np = (np.load(os.path.join(os.getcwd(), 'files/ex5.npy'))).astype(np.uint8)
        ref_img = imageio.imread(os.path.join(os.getcwd(), 'files/ex5_walk.png'))
        ref_img = ref_img.astype(np.uint8)
          
        print('Function 5 IMG: %s' % gen.calc_rmse(img, ref_img))
        self.assertTrue(gen.calc_rmse(img, ref_img) < 1000)
        
        print('Function 5 NPY: %s' % gen.calc_rmse(img, ref_np))
        self.assertTrue(gen.calc_rmse(img, ref_np) < 1000)
        
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
         
        self.assertTrue(np.amax(gen.normalize(gen.generate_img(), 8)) < pow(2, 8))
        self.assertTrue(np.amin(gen.normalize(gen.generate_img(), 8)) >= 0)
