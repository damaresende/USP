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
from math import sqrt, floor


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


class IMGFiltering:
    def normalize(self, img, B):
        '''
        Normalizes image in values between 0 and 2^B - 1
        
        @param img: 2D numpy array to normalize
        @param B: number of bits to define normalization maximum
        '''
        imax = np.max(img)
        imin = np.min(img)
        
        return (img-imin)/(imax-imin) * (pow(2, B) - 1)  
    
    def calc_rmse(self, img1, img2):
        '''
        Computes RMSE in between two images
        
        @param img1: 2D numpy array
        @param img2: 2D numpy array
        @return integer with RMSE value
        '''
        return round(sqrt(sum(sum(np.multiply((img1 - img2), (img1 - img2))))), 4)


class LimiarFilter(IMGFiltering):
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
        optimum_t = self.find_optimum_threshold(img)
        
        new_img = np.copy(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] > optimum_t:
                    new_img[i][j] = 1
                else:
                    new_img[i][j] = 0
        
        return new_img

    def find_optimum_threshold(self, img):
        '''
        Updates the threshold until an optimum value is found
        
        @param img: image to check pixel values
        @return optimum value for the threshold
        '''
        optimum_t = self.threshold
        
        while True: 
            g1_sum = 0.0
            g1_count = 0.0
            
            g2_sum = 0.0
            g2_count = 0.0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j] > optimum_t:
                        g1_sum += img[i][j]
                        g1_count += 1
                    else:
                        g2_sum += img[i][j]
                        g2_count += 1
                        
            new_t = (g1_sum/g1_count + g2_sum/g2_count) / 2.0
            if new_t - optimum_t < 0.5:
                return new_t
            else:
                optimum_t = new_t
        return optimum_t
    

class Filter1D(IMGFiltering):
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
        padding = floor(self.size / 2)
        img_1D = self.create_circular_array(img)
        result = np.zeros((img.shape[0] * img.shape[1],))
        
        for k in range(padding, img.shape[0] * img.shape[1] + 1):
            result[k-1] = sum(np.multiply(self.weights, img_1D[k-1:k+self.size-1]))
            
        return np.reshape(result, img.shape)
        
    def create_circular_array(self, img):
        '''
        Transform a 2D image into a 1D array and pads the borders with values of the circular array
        
        @param img: image to transform to 1D
        @return 1D circular array with image
        ''' 
        padding = floor(self.size / 2)
        aux = np.reshape(img, img.shape[0] * img.shape[1])
        img_1D = np.zeros(img.shape[0] * img.shape[1] + 2 * padding, dtype=np.uint8)
        
        img_1D[padding:-padding] = aux
        img_1D[0:padding] = aux[-padding:]
        img_1D[-padding:] = aux[0:padding]
        
        return img_1D


class LimiarFilter2D(IMGFiltering):
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


class MedianFilter2D(IMGFiltering):
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
     
    img = imageio.imread(img_name)
    filt = FilterFactory.init_filter(method)
    filterd_img = filt.apply_filter(img)
    rmse = filt.calc_rmse((filt.normalize(img, 8)).astype('uint8'), (filterd_img).astype('uint8'))
    
    return img, filterd_img, rmse
    

