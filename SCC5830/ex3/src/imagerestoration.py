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
from math import sqrt, floor


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
    
    @staticmethod
    def normalize(img, gmax):
        '''
        Normalizes image in values between 0 and 255
        
        @param img: 2D numpy array to normalize
        '''
        imax = np.max(img)
        imin = np.min(img)
        
        return ((img-imin)/(imax-imin) * gmax)
    
    @staticmethod
    def calc_rmse(img1, img2):
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
    
    def iqr(self, mask):
        mask = np.sort(mask.flatten())
        middle = round(mask.shape[0]/2)
        quart_middle = round(mask.shape[0]/4)
        return mask[middle + quart_middle] - mask[quart_middle]
        
    def calc_disp(self, mask):
        if self.mode == 'average':
            return np.std(mask)
        elif self.mode == 'robust':
            return self.iqr(mask)
        else:
            raise ValueError("Dispersion mode is unknown.")
      
    def calc_centrl(self, mask):
        if self.mode == 'average':
            return np.mean(mask)
        elif self.mode == 'robust':
            return np.median(mask)
        else:
            raise ValueError("Dispersion mode is unknown.")
            
    def restore_img(self, img):
        dispn = img[0:floor(img.shape[0]/6)-1, 0:floor(img.shape[1]/6)-1]
        dispn = self.calc_disp(dispn)
        dispn = 1 if dispn == 0 else dispn
        
        padding = int((self.kernel-1) / 2)
        new_img = np.copy(img)
        
        for i in range(padding, img.shape[0] - padding):
            for j in range(padding, img.shape[1] - padding):
                mask = img[i-padding:i+padding+1, j-padding:j+padding+1]
                centrl = self.calc_centrl(mask)
                displ = self.calc_disp(mask)
                displ = dispn if displ == 0 else displ
                
                if dispn > displ:
                    displ = dispn
                    
                new_img[i][j] = img[i][j] - (self.gamma * dispn / displ) * (img[i][j] - centrl)
                                
        return new_img


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
    
    deg_img = imageio.imread(deg_img_name).astype(np.float)
    
    rest_filter = FilterFactory.init_filter(filter_type)
    restored_img = FilterFactory.normalize(rest_filter.restore_img(deg_img), np.max(deg_img))
      
    ref_img = imageio.imread(ref_img_name)
    rmse = FilterFactory.calc_rmse(ref_img.astype(np.float), restored_img.astype(np.uint8))
    
    return ref_img, deg_img, restored_img, rmse

    
def main():
    _, _, _, rmse = run_restoration();
    print(rmse)

    
if __name__ == '__main__':
    main()    

        