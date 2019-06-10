'''
Image segmentation

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 10, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Image Processing Class (SCC5830)
'''
import imageio
import numpy as np


class IMGSegmentation:
    pass
    
def run_segmentation():
    """
    TO DO
    """
    ipt_img_name = str(input()).rstrip()
    ref_img_name = str(input()).rstrip()
    
    option = int(input())
    nclusters = int(input())
    niterations = int(input())
    seed = int(input())
    
    ipt_img = imageio.imread(ipt_img_name).astype(np.float)
    ref_img = imageio.imread(ref_img_name).astype(np.float)
    
    return ipt_img, ref_img, ref_img, 0

    
def main():
    """
    Runs restoration
    """
    rmse = run_segmentation();
    print(rmse)

    
if __name__ == '__main__':
    main()    

        