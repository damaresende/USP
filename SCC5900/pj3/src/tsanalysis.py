'''
Has methods for Time Series analysis such as the DTW (Dynamic Time Warping)
algorithm and a method to compute the distance between two time series 
based on DTW.  

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 22, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''
import time
import multiprocessing as mp

from SCC5900.pj3.src.dataparser import DataParser


class TSAnalysis:

    def __init__(self, verbose=False):
        '''
        Sets verbose mode
        '''
        self.verbose = verbose
        self.labels = DataParser.get_labels()
        self.x_train, self.y_train = DataParser.get_training_set()
        
    def cost(self, a, b):
        '''
        Calculates cost between two data points
        
        @param a: floating number. Sample of a of time series A
        @param b: floating number. Sample of b of time series B
        @return cost between the two given data points
        '''
        c = a - b
        return c * c
    
    def dtw(self, tsa, tsb):
        '''
        Applies DTW algorithm to calculate the distance between two time series.
        The distance is computed by taking into consideration all data points of
        both series and the minimum cost of the previous data points. This version
        of DTW uses dynamic programming to store the costs calculated in previous
        steps. TSA and TSB can be of different lengths.
        
        @param tsa: array of floating points to represent time series A
        @param tsb: array of floating points to represent time series B 
        @return floating number with distance between time series A and B
        '''
        m = len(tsa)
        n = len(tsb)
        memo = [[float('inf') for _ in range(n)] for _ in range(m)]
        
        memo[0][0] = self.cost(tsa[0], tsb[0])
        
        for j in range(1, n):
            memo[0][j] = self.cost(tsa[0], tsb[j]) + memo[0][j-1]
    
        for i in range(1, m):
            memo[i][0] = self.cost(tsa[i], tsb[0]) + memo[i-1][0]
             
        for i in range(1, m):
            for j in range(1, n):
                memo[i][j] = self.cost(tsa[i], tsb[j]) + min(memo[i - 1][j - 1], 
                                                             memo[i][j - 1], 
                                                             memo[i - 1][j])

        return memo[m-1][n-1]
    
    def dtw_constrained(self, tsa, tsb, w, k):
        '''
        Applies DTW algorithm to calculate the distance between two time series.
        The distance is computed by taking into consideration all data points of
        both series and the minimum cost of the previous data points. This version
        of DTW reduces the search space of the memoization matrix based on a window
        w, which constrains the values to the w diagonal rows of the matrix. TSA and 
        TSB can be of different lengths.
        
        @param tsa: array of floating points to represent time series A
        @param tsb: array of floating points to represent time series B 
        @param w: window length
        @return floating number with distance between time series A and B
        '''
        m = len(tsa)
        n = len(tsb)
        memo = [[float('inf') for _ in range(n)] for _ in range(m)]
        
        memo[0][0] = self.cost(tsa[0], tsb[0])
        w = max(w, abs(n-m))
      
        for i in range(1, m):
            for j in range(max(1, i-w), min(n, i+w)):
                memo[i][j] = self.cost(tsa[i], tsb[j]) + min(memo[i - 1][j - 1], 
                                                             memo[i][j - 1], 
                                                             memo[i - 1][j])

        return (memo[m-1][n-1], k)
    
    def dtw_constrained_plus(self, tsa, tsb, w, k):
        '''
        Applies DTW algorithm to calculate the distance between two time series.
        The distance is computed by taking into consideration all data points of
        both series and the minimum cost of the previous data points. This version
        of DTW reduces the search space of the memoization matrix based on a window
        w, which constrains the values to the w diagonal rows of the matrix. Also
        reduces the size of memo matrix. TSA and TSB must have equal lengths.
        
        @param tsa: array of floating points to represent time series A
        @param tsb: array of floating points to represent time series B 
        @param w: window length
        @return floating number with distance between time series A and B
        '''
        m = len(tsa)
        n = len(tsb)
        memo = [[float('inf') for _ in range(n)] for _ in range(2)]
        
        p = 0
        c = 1
        
        memo[0][0] = self.cost(tsa[0], tsb[0])
        w = max(w, abs(n-m))
      
        for i in range(1, m):
            for j in range(max(1, i-w), min(n, i+w)):
                memo[c][j] = self.cost(tsa[i], tsb[j]) + min(memo[p][j - 1], 
                                                             memo[c][j - 1], 
                                                             memo[p][j])
            # swap
            tmp = c
            c = p
            p = tmp
        return (memo[1][n-1], k)
    
    def one_nn(self, test_sample, x, w):
        '''
        Applies 1-NN algorithm to find the best fitting class for a given time series.
        1-NN compares the given instance with all instances of the training set
        and takes the distance between the two time series based on the DTW algorithm. Once
        all training samples are verified and the minimum distance is defined, the label of 
        the test instance being evaluated is set.
        
        @param test_sample: array of floats with time series values
        @return integer with the class of the given time series
        '''
        # Searches for nearest neighbor in parallel
        pool = mp.Pool(mp.cpu_count())
        res = pool.starmap_async(self.dtw_constrained, [(test_sample, train_sample, w, k) 
                                  for k, train_sample in enumerate(self.x_train)]).get()
        pool.close()
                
        klass = self.y_train[min(res, key=lambda v : v[0])[1]]
        
        if self.verbose:
            print('Classifying instance %d.' % x, end=' ')
            print('Predicted class is %d: %s' % (klass, self.labels[klass - 1]))
        
        return klass
        
    def calc_accuracy(self, pred_labels, original_labels):
        '''
        Computes the accuracy of prediction between the prediction array and the
        array with original labels. It sums all the labels correctly classified and
        divides it by the total number of instances.
        
        @param pred_labels: array of integers with predicted labels
        @param original_labels: array of integers with test set original labels
        @return floating number with assertion rate
        '''
        accuracy = 0
        for k in range(len(pred_labels)):
            if pred_labels[k] == original_labels[k]:
                accuracy += 1
                
        return accuracy / len(pred_labels)
        
    def print_memo(self, memo):
        '''
        Prints memoization matrix
        
        @param memo: memoization matrix
        '''
        print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in memo]))

    def format_elapsed_time(self, elapsed_time):
        '''
        Computes the elapsed time given a initial time.
        
        @param elapsed_time: float number with elapsed time
        @return elapsed time in a formatted string
        '''
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return '{:0>2}h {:0>2}min {:05.2f}sec'.format(int(hours), int(minutes), seconds)

# Initializes objects
ts = TSAnalysis(verbose=True) # classifies series
x_test, y_test = DataParser.get_test_set() # retrieves test set and labels

init_time = time.time() # starts timer
klass = [ts.one_nn(sample, x, 20) for x, sample in enumerate(x_test)]

time_elapsed = ts.format_elapsed_time(time.time() - init_time) # formats elapsed time string

print('Total Elapsed time is %s' % time_elapsed)
print('Classification Accuracy: %f %%' % (ts.calc_accuracy(klass, y_test) * 100))