'''
Created on Oct 16, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project II: CNN
@author: Damares Resende

Has methods to build a CNN model
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf

class CNNModel():
    def __init__(self, img_rows, img_cols, num_classes, n_channels, 
                                    input_shape = None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_classes = num_classes
        self.n_channels = n_channels
        
        if input_shape:
            self.input_shape = input_shape
        else:
            self.input_shape = (self.img_rows, self.img_cols, self.n_channels)
        
    def build_model(self, model = None, shape = None):            
        if model:
            self.model = model
        elif shape:
            self.model = tf.keras.Sequential()
            for idx in range(len(shape['conv']['filters'])):
                self._add_conv_layer(shape['conv'], idx)
            
            self.model.add(tf.keras.layers.Flatten())
            for idx in range(len(shape['dense']['size'])):
                self._add_dense_layer(shape['dense'], idx)
            
            self.model.add(tf.keras.layers.Dense(self.num_classes,
                                                 activation='softmax'))
            self.model.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.RMSprop(lr=0.0025),
                          metrics=['accuracy'])

        else:
            raise Exception('Could not build model. You must either enter' + 
                ' an model already built or the shape of the model to build.')

    def fit(self, x_train, y_train, x_val, y_val, batch_size, epochs, 
                                            class_distribution, verbose = 1): 
        return self.model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = verbose,
                    validation_data = (x_val, y_val),
                    class_weight = self._class_weights(class_distribution))
        
    def evaluate(self, x_test, y_test, verbose = 0):
        return self.model.evaluate(x_test, y_test, verbose=verbose)
    
    def save_model(self, path_,overwrite = True):
        tf.keras.models.save_model(self.model, path_,
                                   overwrite=overwrite,
                                   include_optimizer=True)

    def _add_conv_layer(self, layer, idx):
            self.model.add(tf.keras.layers.Conv2D(layer['filters'][idx], 
                            kernel_size = layer['kernel'][idx], 
                            activation='relu',
                            input_shape = self.input_shape))
            if layer['pooling'][idx]:
                self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
	    if layer['dropout'][idx] > 0:
		self.model.add(tf.keras.layers.Dropout(layer['dropout'][idx]))
            
    def _add_dense_layer(self, layer, idx):
        self.model.add(tf.keras.layers.Dense(layer['size'][idx],
                                             activation='relu'))
	if layer['dropout'][idx] > 0:
                self.model.add(tf.keras.layers.Dropout(layer['dropout'][idx]))
            
    def _class_weights(self, class_distribution):
        weights = (class_distribution - class_distribution.min()) \
                    / (class_distribution.max() - class_distribution.min())
        
        return {idx: value for idx, value in enumerate(2 - weights)}
