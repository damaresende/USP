'''
Retrieves basic information about the Animals With Attributes 2 dataset. The
data retrieved includes some of the possible classes and attributes.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 28, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import numpy as np
import pandas as pd
from os.path import join
from .logwriter import Logger, MessageType


class AnnotationsParser():
    
    def __init__(self, base_path):
        '''
        Initialization
        
        @param base_path: string that points to path where the base data files are
        '''
        self.base_path = base_path
        
    def get_labels(self):
        '''
        Retrieves the labels available for objects in Animals with Attributes 2 data set
        
        @return list of strings with available labels
        '''
        try:
            file_path = join(self.base_path, 'AwA2-classes.txt')
            with open(file_path) as f:
                labels = {int(line.split()[0]): line.split()[1] for line in f.readlines()}
                
            return labels
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return []
    
    def get_predicates(self):
        '''
        Retrieves the attributes available for objects in Animals with Attributes 2 data set
        
        @return list of strings with available predicates
        '''
        try:
            file_path = join(self.base_path, 'AwA2-predicates.txt')
            with open(file_path) as f:
                predicates = [line.split()[1] for line in f.readlines()]
                
            return predicates
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return []
        
    def get_attributes(self):
        '''
        Retrieves data frame with object labels and corresponding attributes
        
        @return pandas data frame with 12 labels and 85 corresponding attributes
        '''
        try:
            file_path = join(self.base_path, 'AwA2-predicate-matrix.txt')
            
            with open(file_path) as f:
                matrix = np.zeros((12, 85), dtype=np.float32)
                
                for i, line in enumerate(f.readlines()):
                    for j, value in enumerate(line.split()):
                        matrix[i,j] = float(value)
                
            predicates = pd.DataFrame(data=matrix,
                                index=self.get_labels(),
                                columns=self.get_predicates())
  
            real_columns = ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow',
                            'patches', 'spots', 'stripes', 'furry', 'hairless', 'toughskin', 
                            'bulbous', 'bipedal', 'quadrupedal', 'longleg', 'longneck', 'flippers',
                            'hands', 'paws', 'tail', 'horns']
            real_predicates = pd.DataFrame(index=self.get_labels(), columns=real_columns)
            
            for col in real_columns:
                real_predicates[col] = predicates[col]
            
            return real_predicates
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return None
