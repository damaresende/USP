'''
Created on Nov 18, 2018

Universidade de Sao Paulo - USP Sao Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Project III: Adaptative PCA
@author: Damares Resende
'''

import utils
from NNDataset import NNData

x, y = NNData.get_wine_data()

#### Adaptative PCA ###
components = utils.adaptative_pca(x, 0.001, 100, 3, 10, standardize=True)
utils.plot_data(components, y, 'adaptative')
x_train, y_train, x_test, y_test = NNData.stratified_split(x, y, 0.2)
test_loss_0, test_acc_0 = utils.run_wine_mlp(x_train, y_train, x_test, y_test, 
                                             'adp_pca', 2000, plot=True)

#### No PCA ####
x = NNData.normalize(x)
x_train, y_train, x_test, y_test = NNData.stratified_split(x, y, 0.2)
test_loss_1, test_acc_1 = utils.run_wine_mlp(x_train, y_train, x_test, y_test, 
                                             'no_pca', 2000, plot=True)
 
### With PCA ###
components, _, _ = utils.calc_pca(x, 3, standardize=True)
utils.plot_data(components, y, 'normal')
x_train, y_train, x_test, y_test = NNData.stratified_split(components, y, 0.2)
test_loss_2, test_acc_2 = utils.run_wine_mlp(x_train, y_train, x_test, y_test, 
                                             'nml_pca', 2000, plot=True)


print('\nResults for classification with adaptative PCA')
print('Test loss: %s' % str(test_loss_0))
print('Test accuracy: %s' % str(test_acc_0))

print('\nResults for classification without PCA')
print('Test loss: %s' % str(test_loss_1))
print('Test accuracy: %s' % str(test_acc_1))
 
print('\nResults for classification with normal PCA')
print('Test loss: %s' % str(test_loss_2))
print('Test accuracy: %s' % str(test_acc_2))