'''
Created on Oct 25, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project II: CNN
@author: Damares Resende

Load the model previously trained and test it against the new images created.
'''

from cnn_data import CNNData
from cnn_params import CNNParams
from cnn_model import CNNModel

params = CNNParams()
data = CNNData(params.img_height, params.img_width)
print('Loading data...\n')
x_test, y_test, _ = data.getTestData()
x_test, y_test = data.prepare_data(x_test, y_test)

try:
    print('Loading model from %s\n' % params.model_path)
    cnn = CNNModel(params.img_height, params.img_width, 
                   params.n_classes, params.n_channels)
    cnn.build_model(shape = params.model_shape)
    
    cnn.model.load_weights(params.model_path)
    cnn.model.summary()
    
    metrics = cnn.evaluate(x_test, y_test)
    print('Loss: %.4f' % (metrics[0]))
    print('Accuracy: %.4f' % (metrics[1]))
except OSError:
    print('Could not load model from %s' % params.model_path)

print('bye o/')
