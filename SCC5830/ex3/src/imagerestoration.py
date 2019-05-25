'''
Image restoration

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 11, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import imageio
import numpy as np
from enum import Enum
from math import sqrt, floor
from scipy.fftpack import fftn, ifftn, ifftshift


class NoiseType(Enum):
    """
    Type of noises to add to image
    """
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
        """
        Creates and saves a noisy image based on an input image
        
        @param img_name_in: string with input image
        @param img_name_out: string with output image
        @param noise_type: type of noise to add to image
        @param kwargs: arguments to be used to generate the noise
        """
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
            window = np.ones((7,7))
            window /= np.sum(window)
            from scipy.signal import convolve2d
            blured_img = convolve2d(img, window, mode="same", boundary="symm")
            imageio.imwrite(img_name_out, blured_img.astype(np.uint8))

       
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
        """
        Initializes main parameters
        """
        self.gamma = float(input())
        self.kernel = int(input())
        self.mode = str(input()).strip()
    
    def iqr(self, mask):
        """
        Computes the interquartile range
        
        @param mask: set of reference pixels
        """
        mask = np.sort(mask.flatten())
        middle = round(mask.shape[0]/2)
        quart_middle = round(mask.shape[0]/4)
        return mask[middle + quart_middle] - mask[quart_middle]
        
    def calc_disp(self, mask):
        """
        Calculates the dispersion measure
        
        @param mask: set of reference pixels
        """
        if self.mode == 'average':
            return np.std(mask)
        elif self.mode == 'robust':
            return self.iqr(mask)
        else:
            raise ValueError("Dispersion mode is unknown.")
      
    def calc_centrl(self, mask):
        """
        Calculates the centrality measure
        
        @param mask: set of reference pixels
        """
        if self.mode == 'average':
            return np.mean(mask)
        elif self.mode == 'robust':
            return np.median(mask)
        else:
            raise ValueError("Dispersion mode is unknown.")
            
    def restore_img(self, img):
        """
        Removes noise from the given image based on the dispersion of a sample of the image
        and the centrality measure of each pixel.
        
        @param img: numpy array with image to be filtered
        """
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
        """
        Initializes main parameters
        """
        self.gamma = float(input())
        self.kernel = int(input())
        self.sigma = float(input())

    def restore_img(self, img):
        """
        Deblurs a given image
        
        @param img: numpy array with image to be filtered
        """
        g_u = fftn(img)
        h_u = fftn(self.gaussian_filter(img, self.kernel, self.sigma))
        h_u_abs = np.abs(h_u)
        p_u_abs = np.abs(fftn(self.get_laplacian_operator(h_u)))
        
        degradation = (np.divide(np.conj(h_u).transpose(), np.multiply(h_u_abs, h_u_abs) 
                                 + self.gamma * np.multiply(p_u_abs, p_u_abs)))
    
        return np.real(ifftshift(ifftn(np.multiply(degradation, g_u))))
    
    def gaussian_filter(self, ref, k=3, sigma=1.0):
        """
        Produces a Gaussian filter and expands it to have the same size as the image to be filtered
        
        @param ref: image to be filtered
        @param k: mask initial size
        @param sigma: adjustment factor
        """
        arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
        x, y = np.meshgrid(arx, arx)
        filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
        filt = filt / np.sum(filt)
        
        padding = ref.shape[0]//2-filt.shape[0]//2
        return np.pad(filt, (padding,padding-1), "constant",  constant_values=0)
        
        
    def get_laplacian_operator(self, ref):
        """
        Builds a Laplacian operator to convolve it with the image. The initial operator is expanded to fit the
        size of the image.
        
        @param ref: image to be filtered
        """
        operator = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        padding = ref.shape[0]//2-operator.shape[0]//2
        
        return np.pad(operator, (padding,padding-1), "constant",  constant_values=0)
    
def run_restoration():
    """
    Restores a corrupted image computes the rmse to compare it with the original one
    """
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
    """
    Runs restoration
    """
    _, _, _, rmse = run_restoration();
    print(rmse)

    
if __name__ == '__main__':
    main()    

        