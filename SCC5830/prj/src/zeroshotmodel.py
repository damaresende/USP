'''
Classifies unknown classes based on semantic data

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 29, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout

from src.featuresparser import FeaturesParser
from src.annotationsparser import AnnotationsParser

 
def main():
    # Get data
    project_path = os.path.join(os.getcwd().split('SCC5830')[0], 'SCC5830')
    parser = FeaturesParser(os.path.join(project_path, 'data'))
    data = parser.get_data()
    
    # Build encoding model
    model = Sequential()
    model.add(Dense(2560, activation='relu', input_shape=(data['x_train_vis'].shape[1],)))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(data['x_train_sem'].shape[1], activation='linear'))
 
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mean_absolute_error'])
     
    print(model.summary())
    
    history = model.fit(data['x_train_vis'], data['x_train_sem'],
              batch_size=256,
              epochs=150,
              verbose=1,
              validation_split=0.2)
    
    # Encode features
    prediction = model.predict(data['x_test_vis'])
     
    # Find 1-NN best match
    parser = AnnotationsParser(os.path.join(project_path, 'data'))
    att_map = parser.get_attributes()
    
    classification = np.zeros((prediction.shape[0],))
    for idx, pred in enumerate(prediction):
        min_dist = float('inf')
        
        for k in att_map.index:
            dist = np.linalg.norm(list(att_map.loc[k]) - pred, axis=0)
            
            if dist < min_dist:
                classification[idx] = k
                min_dist = dist

    # Check classification
    prediction = ''
    count = 0
    for idx, y in enumerate(data['y_test']):
        prediction += 'Instance %d. True Class %d. Pred Class %d.\n' % (idx, y, classification[idx])
        if y == classification[idx]:
            count += 1
    
    print('Classification accuracy is %s %%' % str(round(count * 100 / len(data['y_test']), 4)))
    # Final Classification accuracy is 3.7037 % - much worst than random
        
    # Plot Results
    plt.figure(figsize=(16,6))
    plt.rcParams.update({'font.size': 10})
    
    plt.subplot(131)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.legend(['loss', 'val_loss'])

    plt.subplot(132)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Acc')
    plt.legend(['acc', 'val_acc'])
    
    plt.subplot(133)
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Abs. Error')
    plt.legend(['mae', 'val_mae'])
    
    plt.savefig(os.path.join(os.path.join(project_path, 'results'), 'performance.png'))
    
    # Save prediction
    with open(os.path.join(os.path.join(project_path, 'results'), 'prediction.txt'), 'w') as f:
        f.write(prediction)
        
    # Save results
    with open(os.path.join(os.path.join(project_path, 'results'), 'results.txt'), 'w') as f:
        f.write('loss: ' + ','.join([str(round(v, 4)) for v in history.history['loss']]) + '\n')
        f.write('val_loss: ' + ','.join([str(round(v, 4)) for v in history.history['val_loss']]) + '\n')
        f.write('acc: ' + ','.join([str(round(v, 4)) for v in history.history['acc']]) + '\n')
        f.write('val_acc: ' + ','.join([str(round(v, 4)) for v in history.history['val_acc']]) + '\n')
        f.write('mae: ' + ','.join([str(round(v, 4)) for v in history.history['mean_absolute_error']]) + '\n')
        f.write('val_mae: ' + ','.join([str(round(v, 4)) for v in history.history['val_mean_absolute_error']]) + '\n')
        f.write('classification: ' + ','.join([str(int(v)) for v in classification]) + '\n')
    
    # Save model
    model.save_weights(os.path.join(os.path.join(project_path, 'results'), 'model.h5'))
        
    # Save data
    with open(os.path.join(os.path.join(project_path, 'results'), 'x_test_vis.txt'), 'w') as f:
        for row in data['x_test_vis']:
            f.write(', '.join(map(str, list(row))) + '\n')
            
    with open(os.path.join(os.path.join(project_path, 'results'), 'x_test_sem.txt'), 'w') as f:
        for row in data['x_test_sem']:
            f.write(', '.join(map(str, list(row))) + '\n')
    
    with open(os.path.join(os.path.join(project_path, 'results'), 'y_test.txt'), 'w') as f:
        f.write('\n'.join(map(str, list(data['y_test']))))
        
if __name__ == '__main__':
    main()