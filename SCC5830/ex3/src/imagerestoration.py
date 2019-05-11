'''
Image restoration

@author: Damares Resende
@contact: damaresresende@usp.br
@since: MAy 11, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import imageio
import numpy as np
from enum import Enum
from math import sqrt


class NoiseType(Enum):
    UNIFORM = 1
    GAUSSIAN = 2
    IMPULSIVE = 3
    BLURED = 4
        

class IMGNoiseGenerator:
    @staticmethod
    def uniform_noise(size, prob=0.1):
        '''
        Generates a matrix with uniform noise in the range [0-255] to be added to an image
        
        @param size: tuple defining the size of the noise matrix 
        @param prob: probability for the uniform noise generation 
        @return matrix with uniform noise to be added to image
        '''
        
        levels = int((prob * 255) // 2)
        noise = np.random.randint(-levels, levels, size)
        
        return noise
    
    @staticmethod
    def gaussian_noise(size, mean=0, std=0.01):
        '''
        Generates a matrix with Gaussian noise in the range [0-255] to be added to an image
        
        @param size: tuple defining the size of the noise matrix 
        @param mean: mean of the Gaussian distribution
        @param std: standard deviation of the Gaussian distribution, default 0.01
        @return matrix with Gaussian noise to be added to image
        '''
        noise = np.multiply(np.random.normal(mean, std, size), 255)
        
        return noise
    
    @staticmethod
    def impulsive_noise(image, prob=0.1):
        '''
        Returns image with impulsive noise (0 and/or 255) to replace pixels in the image with some probability
        
        @param image: input image
        @param prob: probability for the impulsive noise generation 
        @param mode: type of noise, 'salt', 'pepper' or 'salt_and_pepper' (default)
        @return noisy image with impulsive noise
        '''
    
        noise = np.array(image, copy=True)
        for x in np.arange(image.shape[0]):
            for y in np.arange(image.shape[1]):
                rnd = np.random.random()
                if rnd < prob:
                    rnd = np.random.random()
                    if rnd > 0.5:
                        noise[x,y] = 255
                    else:
                        noise[x,y] = 0
        
        return noise

    @staticmethod
    def create_noisy_image(img_name_in, img_name_out, noise_type, **kwargs):
        img = imageio.imread(img_name_in)
        
        if noise_type == NoiseType.UNIFORM:
            noise = IMGNoiseGenerator.uniform_noise(img.shape, kwargs.get("prob", 0.1))
            imageio.imwrite(img_name_out, img + noise)
        elif noise_type == NoiseType.GAUSSIAN:
            noise = IMGNoiseGenerator.gaussian_noise(img.shape, kwargs.get("mean", 0), kwargs.get("std", 0.01))
            imageio.imwrite(img_name_out, img + noise)
        elif noise_type == NoiseType.IMPULSIVE:
            noisy_img = IMGNoiseGenerator.impulsive_noise(img, kwargs.get("prob", 0.1))
            imageio.imwrite(img_name_out, noisy_img)
        elif noise_type == NoiseType.BLURED:
            pass

       
class FilterFactory:
    @staticmethod
    def init_filter(method):
        '''
        Returns an object with the filter implementation specified
        
        @return filter
        '''
        if method == 1:
            return DenoisingFilter()
        if method == 2:
            return DebluringFilter()
        
    def normalize(self, img):
        '''
        Normalizes image in values between 0 and 255
        
        @param img: 2D numpy array to normalize
        '''
        imax = np.max(img)
        imin = np.min(img)
        
        return ((img-imin)/(imax-imin) * (pow(2, 8) - 1)).astype(np.int32)
    
    def calc_rmse(self, img1, img2):
        '''
        Computes RMSE in between two images
        
        @param img1: 2D numpy array
        @param img2: 2D numpy array
        @return integer with RMSE value
        '''
        mse = np.sum(np.multiply(img1 - img2, img1 - img2)) / (img1.shape[0] * img1.shape[1])
        return round(sqrt(mse), 4)


class DenoisingFilter:
    def __init__(self):
        self.gamma = float(input())
        self.kernel = int(input())
        self.mode = str(input()).strip()
    
    def restore_img(self):
        pass
    

class DebluringFilter:
    def __init__(self):
        self.gamma = float(input())
        self.kernel = int(input())
        self.sigma = float(input())

    def restore_img(self):
        pass

    
def run_restoration():
    ref_img_name = str(input()).rstrip()
    deg_img_name = str(input()).rstrip()
    filter_type = int(input())
    
    ref_img = imageio.imread(deg_img_name)
    
    rest_filter = FilterFactory.init_filter(filter_type)
    restored_img = rest_filter.restore_img(ref_img)
      
    deg_img = imageio.imread(ref_img_name)
    rmse = rest_filter.calc_rmse(deg_img, rest_filter.normalize(restored_img))
    
    return ref_img, deg_img, restored_img, rmse

    
def main():
    _, _, _, rmse = run_restoration();
    print(rmse)

    
if __name__ == '__main__':
    main()    

        