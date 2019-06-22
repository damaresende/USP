'''
Image segmentation

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 10, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import random
import imageio
import numpy as np
from math import sqrt
from copy import deepcopy


class IMGSegmentation:
    
    @staticmethod
    def img_to_dataset(img, option):
        '''
        Creates a N x K numpy array based on the given image where N is m * n 
        and k is 3 for options 1 and 4, 5 for option 2 and 1 for option 3.
        
        @param img: image to transform to data set
        @param option: integer with type of data set to be built
        @return numpy array with data set shape N x K
        '''
        dim = img.shape[0] * img.shape[1]
        
        if option == 1:    
            dataset = np.zeros((dim, 3))
            dataset[:, 0] = img[:, :, 0].reshape((dim,))
            dataset[:, 1] = img[:, :, 1].reshape((dim,))
            dataset[:, 2] = img[:, :, 2].reshape((dim,))

        elif option == 2:
            dataset = np.zeros((dim, 5))
            dataset[:, 0] = img[:, :, 0].reshape((dim,))
            dataset[:, 1] = img[:, :, 1].reshape((dim,))
            dataset[:, 2] = img[:, :, 2].reshape((dim,))
            
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    dataset[x * img.shape[0] + y, 3] = x
                    dataset[x * img.shape[0] + y, 4] = y
                    
        elif option == 3:
            dataset = 0.299 * img[:, :, 0].reshape((dim,1)) \
                      + 0.587 * img[:, :, 1].reshape((dim,1)) \
                      + 0.144 * img[:, :, 2].reshape((dim,1))
                      
        elif option == 4:
            dataset = np.zeros((dim, 3))
            dataset[:, 0] = 0.299 * img[:, :, 0].reshape((dim,)) \
                            + 0.587 * img[:, :, 1].reshape((dim,)) \
                            + 0.144 * img[:, :, 2].reshape((dim,))
        
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    dataset[x * img.shape[0] + y, 1] = x
                    dataset[x * img.shape[0] + y, 2] = y
        return dataset
        
    @staticmethod
    def calc_distance(a, b, ax=1):
        '''
        Calculates the Euclidean distance between a and b
        
        @param a: coordinates of point A
        @param b: coordinates of point B
        @return float with distance value
        '''
        return np.linalg.norm(a - b, axis=ax)
    
    @staticmethod
    def normalize(img):
        '''
        Normalizes image in values between 0 and 255
        
        @param img: 2D numpy array to normalize
        '''
        imax = np.max(img)
        imin = np.min(img)
        
        return ((img-imin)/(imax-imin) * 255)
    
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
    
    @staticmethod
    def k_means(k, data, n, seed):
        '''
        Computes the kmeans on the points of the given data set in order to clusterize them
        
        @param k: number of clusters
        @param data: data set
        @param n: stopping criteria. Maximum number of interactions
        @param seed: seed to pick the initial centroids
        @return 1D numpy array with clusters classification
        '''
        random.seed(seed)
        clusters = np.zeros((data.shape[0]), dtype=np.uint8)
        
        centroids = np.zeros((k, data.shape[1]))
        for x, id_ in enumerate(np.sort(random.sample(range(0, data.shape[0]), k))):
            centroids[x, :] = data[id_]
        
        for _ in range(n):
            for x, pixel in enumerate(data):
                dist = IMGSegmentation.calc_distance(pixel, centroids)
                clusters[x] = np.argmin(dist)
                
            centroids_bkp = deepcopy(centroids)
            for i in range(k):
                points = [data[j, :] for j in range(data.shape[0]) if clusters[j] == i]
                centroids[i, :] = np.mean(points, axis=0)

            if IMGSegmentation.calc_distance(centroids, centroids_bkp, None) == 0:
                break
            
        return clusters
    

def run_segmentation():
    """
    Segments an image based on its pixel distribution. First it transforms the image in
    a data set according to the given option. The options take into consideration the image's
    RGB pixel's values, position and luminance. Then it applies k-means algorithm to the data
    set to form the k clusters asked. 
    """
    ipt_img_name = str(input()).rstrip()
    ref_img_name = str(input()).rstrip()
    
    option = int(input())
    nclusters = int(input())
    niterations = int(input())
    seed = int(input())
    
    ipt_img = imageio.imread(ipt_img_name).astype(np.float32)
    ref_img = np.load(ref_img_name).astype(np.float32)
    
    dataset = IMGSegmentation.img_to_dataset(ipt_img, option)
    clusters = IMGSegmentation.k_means(nclusters, dataset, niterations, seed)
    cmp_img = clusters.reshape((ipt_img.shape[0], ipt_img.shape[1])).astype(np.uint8)
    cmp_img = IMGSegmentation.normalize(cmp_img)
    rmse = IMGSegmentation.calc_rmse(ref_img, cmp_img)
    
    return ipt_img, ref_img, cmp_img, rmse

    
def main():
    """
    Runs segmentation
    """
    _, _, _, rmse = run_segmentation();
    print(rmse)

    
if __name__ == '__main__':
    main()    

        