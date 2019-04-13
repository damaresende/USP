'''
Image filtering

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 13, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import imageio
import numpy as np


class FilterFactory:
    @staticmethod
    def init_filter(method):
        '''
        Returns an object with the filter implementation specified
        
        @return filter
        '''
        if method == 1:
            return LimiarFilter()
        if method == 2:
            return Filter1D()
        if method == 3:
            return LimiarFilter2D()
        if method == 4:
            return MedianFilter2D()


class LimiarFilter:
    def __init__(self):
        '''
        Reads inputs
        '''
        self.threshold = int(input()) 
        
    def apply_filter(self, img):
        '''
        Applies filter to image
        
        @param img: image to be filtered
        @return filtered image
        '''
        pass


class Filter1D:
    def __init__(self):
        '''
        Reads inputs
        '''
        self.size = int(input())
        self.weights = np.zeros((self.size,))
        
        for i, value in enumerate(input().split(' ')):
            self.weights[i] = int(value)
            
    def apply_filter(self, img):
        '''
        Applies filter to image
        
        @param img: image to be filtered
        @return filtered image
        '''
        pass


class LimiarFilter2D:
    def __init__(self):
        '''
        Reads inputs
        '''
        self.size = int(input())
        self.weights = np.zeros((self.size, self.size))
        
        for i in range(self.size):
            for j in range(self.size):
                self.weights[i][j] = int(input())
        
        self.threshold = int(input())
        
    def apply_filter(self, img):
        '''
        Applies filter to image
        
        @param img: image to be filtered
        @return filtered image
        '''
        pass


class MedianFilter2D:
    def __call__(self):
        '''
        Reads inputs
        '''
        self.size = int(input())

    def apply_filter(self, img):
        '''
        Applies filter to image
        
        @param img: image to be filtered
        @return filtered image
        '''
        pass

def run_filtering():
    img_name = str(input()).rstrip()
    method = int(input())
     
    filt = FilterFactory.init_filter(method)
    return filt.apply_filter(imageio.imread(img_name))
    

