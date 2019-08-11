'''
Loads model and classifies data

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Jun 24, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

from src.annotationsparser import AnnotationsParser


project_path = os.path.join(os.getcwd().split('SCC5830')[0], 'SCC5830')


def create_model(data_vis, data_sem):
    model = Sequential()
    model.add(Dense(2560, activation='relu', input_shape=(data_vis.shape[1],)))
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
    model.add(Dense(data_sem.shape[1], activation='linear'))
 
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mean_absolute_error'])
    
    return model

def load_data():
    data_vis = []
    data_sem = []
    
    with open(os.path.join(os.path.join(project_path, 'results'), 'x_test_vis.txt')) as f:
        for line in f.readlines():
            if line != '\n' and line != '':
                data_vis.append([float(v) for v in line.split(',')])
                
    with open(os.path.join(os.path.join(project_path, 'results'), 'x_test_sem.txt')) as f:
        for line in f.readlines():
            if line != '\n' and line != '':
                data_sem.append([float(v) for v in line.split(',')])
               
    with open(os.path.join(os.path.join(project_path, 'results'), 'y_test.txt')) as f:
        labels = [float(v) for v in f.readlines()]
    
    return np.array(data_vis), np.array(data_sem), np.array(labels)


data_vis, data_sem, lbs = load_data()
model = create_model(data_vis, data_sem)
model.load_weights(os.path.join(os.path.join(project_path, 'results'), 'model.h5'))
prediction = model.predict(data_vis)

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
count = 0
for idx, y in enumerate(lbs):
    print('Instance %d. True Class %d. Pred Class %d.' % (idx, y, classification[idx]))
    if y == classification[idx]:
        count += 1

print('Classification accuracy is %s %%' % str(round(count * 100 / len(lbs), 4)))