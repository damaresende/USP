'''
Retrieves features images extracted with ResNet101. Each feature vector has
2048 features.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 28, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import numpy as np
from os.path import join
from keras.utils import normalize

from .logwriter import Logger, MessageType
from .annotationsparser import AnnotationsParser


class FeaturesParser():
    
    def __init__(self, features_path):
        '''
        Initialization
        
        @param base_path: string that points to path where the features data files are
        '''
        self.features_path = features_path
        
    def get_labels(self):
        '''
        Retrieves the labels of each image in Animals with Attributes 2 data set
        
        @return numpy array of integers with labels
        '''
        try:
            file_path = join(self.features_path, 'AwA2-labels.txt')
            
            with open(file_path) as f:
                lines = f.readlines()
                labels = np.zeros((len(lines),), dtype=np.int32)
                
                for idx, line in enumerate(lines):
                    labels[idx] = int(line)
                
            return labels
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return None
        
    def get_visual_features(self, norm=False, norm_axis=1):
        '''
        Retrieves features extracted by ResNet101
        
        @param norm: normalize features
        @return numpy array with features for images in AwA2 data set
        '''
        try:
            file_path = join(self.features_path, 'AwA2-features.txt')
            with open(file_path) as f:
                lines = f.readlines()
                features = np.zeros((len(lines), 2048), dtype=np.float32)
                
                for i, line in enumerate(lines):
                    for j, value in enumerate(line.split()):
                        features[i, j] = float(value)
                
            if norm:
                Logger().write_message('Normalizing visual features.', MessageType.INF)
                return normalize(features, order=2, axis=norm_axis)
        
            return features
        except FileNotFoundError:
            Logger().write_message('File %s could not be found.' % file_path, MessageType.ERR)
            return None
    
    def get_semantic_features(self, norm=False, norm_axis=1):
        '''
        Retrieves semantic features based on annotations
        
        @param norm: normalize features
        @return numpy array with features for images in AwA2 data set
        '''
        ann_parser = AnnotationsParser(self.features_path)
        att_map = ann_parser.get_attributes()
        available_labels = self.get_labels()
        
        features = np.zeros((available_labels.shape[0], 24), dtype=np.float32)
        for idx, label in enumerate(available_labels):
            features[idx, :] = att_map.loc[label].values
        
        if norm:
            Logger().write_message('Normalizing semantic features.', MessageType.INF)
            return normalize(features, order=2, axis=norm_axis)
        return features

    def get_data(self):
        '''
        Splits the visual and semantic data into training and test sets and saves 
        it into a dictionary. The labels defining the training and test sets are
        described in AwA2-train-test-split.txt.
        
        @return dictionary with data split
        '''
        key = ''
        lbs_dict = dict()
        
        with open(os.path.join(self.features_path, 'AwA2-train-test-split.txt')) as f:
            for line in f.readlines():
                if line == '\n' or line == '':
                    continue
                
                if line[0].isdigit():
                    lbs_dict[key].append(line.split())
                else:
                    key = line.strip()
                    lbs_dict[key] = []
            
        sem_fts = self.get_semantic_features()
        vis_fts = self.get_visual_features()
        labels = self.get_labels()
        
        test_mask = [False] * labels.shape[0]
        train_mask = [False] * labels.shape[0]
        train_labels = [int(lb[0]) for lb in lbs_dict['TRAIN']]
        test_labels = [int(lb[0]) for lb in lbs_dict['TEST']]
        
        for idx, lb in enumerate(labels):
            if lb in train_labels:
                train_mask[idx] = True
            elif lb in test_labels:
                test_mask[idx] = True
        
        data = dict()
        data['x_train_vis'] = vis_fts[train_mask, :]
        data['x_train_sem'] = sem_fts[train_mask, :]
        data['x_test_vis'] = vis_fts[test_mask, :]
        data['x_test_sem'] = sem_fts[test_mask, :]
        data['y_train'] = labels[train_mask]
        data['y_test'] = labels[test_mask]
        
        return data