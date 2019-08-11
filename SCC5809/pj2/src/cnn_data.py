'''
Created on Oct 16, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project II: CNN
@author: Damares Resende

Has methods to load and treat QIDER data.
'''

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
from os.path import isfile, join
from keras import backend as K

class CNNData():
    def __init__(self, img_hight, img_width, n_channels = 1):
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'negative', \
                   'neutral', 'sad', 'surprise']
        self.height = img_hight
        self.width = img_width
        self.n_channels = n_channels
        
    def getTrainData(self):
        return self._load_images('data/QIDER/train/', 'Training')
    
    def getValData(self):
        return self._load_images('data/QIDER/val/', 'Validation')
    
    def getTestData(self):
        return self._load_images('data/FACES/', 'External')
    
    def _load_images(self, path_, flag_):
        dataset = np.zeros((0, self.height, self.width, 1))
        class_dist = np.zeros(len(self.classes))
        labels = []
        
        for idx, class_ in enumerate(self.classes):
            dir_ = os.path.join(os.getcwd(), path_, class_)
            img_names = [os.path.join(dir_, im) for im in os.listdir(dir_) \
                         if isfile(join(dir_, im)) and (im.endswith('.jpg') \
                         or im.endswith('.png') or im.endswith('.jpeg'))]
    
        
            images = self._decode_image(img_names)
            print('Loaded images from class: ' + class_ + \
                  '. Shape: ' + str(np.shape(images)))
            
            dataset = np.vstack((dataset, images))
            class_dist[idx] = len(images)
            labels = labels + [idx for _ in range(np.shape(images)[0])]
            
        print('\n'+flag_+' dataset is ready. Shape: '+str(np.shape(dataset)))
        print(flag_+' labels set is ready. Shape: '+str(np.shape(labels))+'\n')
        return dataset.astype(np.uint8), np.array(labels, np.uint8), class_dist
    
    def _decode_image(self, img_names):
        images = []
        graph = tf.Graph()
        with graph.as_default():
            file_name = tf.placeholder(dtype=tf.string)
            file = tf.read_file(file_name)
            image = tf.image.decode_jpeg(file, channels = self.n_channels)
            image = tf.image.resize_images(image, [self.height, self.width])
        
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()   
            for i in range(len(img_names)):
                images.append(session.run(image, \
                                feed_dict={file_name: img_names[i]}))
            session.close()
        return images
    
    def prepare_data(self, x_data, y_data):
        if K.image_data_format() == 'channels_first':
            x_data = x_data.reshape(x_data.shape[0], self.n_channels, \
                                      self.height, self.width)
            self.input_shape = (self.n_channels, self.height, self.width)
        else:
            x_data = x_data.reshape(x_data.shape[0], self.height, \
                                      self.width, self.n_channels)
            self.input_shape = (self.height, self.width, self.n_channels)
            
        x_data = x_data.astype('float32') / 255.0
        y_data = tf.keras.utils.to_categorical(y_data, len(self.classes))
        
        return x_data, y_data
    
    def augment_dataset(self, x, y, class_dist):
        max_img = class_dist.max()
        new_class_dist = [int(v) for v in class_dist]
        y = list(y)
        
        bs = 0
        graph = tf.Graph()
        for i in range(len(self.classes)):
            with graph.as_default():
                naug = max_img - class_dist[i]    
                if naug > 0:
                    naug -= class_dist[i]
                    naug = 0 if naug > 0 else naug
                    flipped_image = tf.image.flip_left_right(
                                x[bs:int(bs+class_dist[i]-naug)])

                    with tf.Session(graph=graph) as session:
                        tf.global_variables_initializer().run()
                        x = np.vstack((x, session.run(flipped_image)))
                        y = y + [i for _ in range(flipped_image.shape[0])]
                        session.close()
                    
                    new_class_dist[i] += int(flipped_image.shape[0])
            bs += int(class_dist[i])
        
        bs = 0
        total = int(sum(class_dist))
        for i in range(len(self.classes)):
            naug = max_img - new_class_dist[i]
            if naug > 0:
                 
                naug = min(naug, class_dist[i])
                rest = min(max_img - new_class_dist[i] - naug, class_dist[i])
                 
                with tf.Session(graph=graph) as session:
                    noise = tf.random_normal(shape=tf.shape(x[0]), mean=0.0, 
                                        stddev=(50)/(255), dtype=tf.float32)
                 
                    indexes = list(range(bs, int(bs + naug))) + \
                                list(range(total, int(total + rest)))
                  
                    noisy_imgs = x[indexes] + noise
                    x = np.vstack((x, session.run(noisy_imgs)))
                    y = y + [i for _ in range(noisy_imgs.shape[0])]
                    session.close()
                 
                new_class_dist[i] += int(noisy_imgs.shape[0])
            total += int(class_dist[i])
            bs += int(class_dist[i])
            
        return x, np.array(y), np.array(new_class_dist)