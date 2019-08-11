'''
Created on Oct 16, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project II: CNN
@author: Damares Resende

Load the dataset, augment it, reshape images, normalize the data, train a 
CNN and save the model.
'''

from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
from cnn_data import CNNData
from cnn_model import CNNModel
from cnn_params import CNNParams
import matplotlib.pyplot as plt

print('Loading data...\n')
params = CNNParams()
data = CNNData(params.img_height, params.img_width)
x_train, y_train, class_dist = data.getTrainData()
x_train, y_train, class_dist = data.augment_dataset(x_train, y_train, class_dist)
print('Class Distribution for augmented set: %s\n' % str(class_dist))
 
x_val, y_val, _ = data.getValData()
x_train, y_train = data.prepare_data(x_train, y_train)
x_val, y_val = data.prepare_data(x_val, y_val)

print('Training model...\n')
cnn = CNNModel(params.img_height, params.img_width, 
               params.n_classes, params.n_channels)
cnn.build_model(shape = params.model_shape)
cnn.model.summary()
history = cnn.fit(x_train, y_train, x_val, y_val,batch_size = params.batch_size, 
                  epochs = params.n_epochs, class_distribution = class_dist)
 

print('Saving model to path %s\n' % params.model_path)
cnn.save_model(params.model_path)
 
print('Saving training charts to path %s\n' % os.path.join(os.getcwd(), 'log'))
fig = plt.figure()
plt.plot([v for v in range(params.n_epochs)], np.array(history.history['loss']))
plt.plot([v for v in range(params.n_epochs)], np.array(history.history['val_loss']))
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig(os.path.join(os.getcwd(), 'log/loss.png'), dpi=fig.dpi)
 
fig = plt.figure()
plt.plot([v for v in range(params.n_epochs)], np.array(history.history['acc']))
plt.plot([v for v in range(params.n_epochs)], np.array(history.history['val_acc']))
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig(os.path.join(os.getcwd(), 'log/acc.png'), dpi=fig.dpi)
 
print('Evaluating...\n')
metrics = cnn.evaluate(x_val, y_val)
print('Loss: %.4f' % (metrics[0]))
print('Accuracy: %.4f\n' % (metrics[1]))
 
print('bye o/')
