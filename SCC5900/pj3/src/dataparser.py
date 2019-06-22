'''
Retrieves labels, training and test data from text files 

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 22, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''


class DataParser:
    
    @staticmethod
    def get_labels():
        '''
        Retrieves a list with the available labels. In total the list must 
        contain 12 labels.
        
        @return list of strings with labels
        '''
        try:
            with open('../data/rotulos.txt') as f:
                labels = [r.split()[1].strip() for r in f.readlines()]
                
            return labels 
        except OSError as e:
            print('ERROR: Could not read data.\n%s' % e)
            return []
    
    @staticmethod
    def get_training_set():
        '''
        Retrieves list training data and training labels. The result must a be a tuple 
        with a list of lists where the outer list has length 240 and the inner list has 
        length X, where X is the number of data points in the time series, and a list of 
        labels of length 240 from values 1 to 12.
        
        @return: tuple with lists of training data and training labels
        '''
        try:
            data = []
            labels = []
            
            with open('../data/treino.txt') as f:
                for line in f.readlines():
                    values = [float(v.strip()) for v in line.split()]
                    data.append(values[1:])
                    labels.append(values[0])

            return data, labels
        
        except OSError as e:
            print('ERROR: Could not read data.\n%s' % e)
            return None   
    
    @staticmethod
    def get_test_set():
        '''
        Retrieves list test data and test labels. The result must a be a tuple with 
        a list of lists where the outer list has length 960 and the inner list has 
        length X, where X is the number of data points in the time series, and a list 
        of labels of length 960 from values 1 to 12.
        
        @return: tuple with lists of training data and training labels
        '''
        try:
            data = []
            labels = []
            
            with open('../data/teste.txt') as f:
                for line in f.readlines():
                    values = [float(v.strip()) for v in line.split()]
                    data.append(values[1:])
                    labels.append(values[0])

            return data, labels
        
        except OSError as e:
            print('ERROR: Could not read data.\n%s' % e)
            return None
