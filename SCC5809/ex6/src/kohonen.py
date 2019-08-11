'''
Created on Nov 25, 2018

@author: damaresresende
'''

from __future__ import print_function

import math
import random
import numpy as np
from matplotlib import pyplot as plt

from dataset import Data


class Neuron:
    def __init__(self, weights):
        self.weights = weights
        self.samples = []
        self.label = -1

class Konehan:
    def __init__(self, data, labels, m, n):
        self.m = m
        self.n = n
        self.data = data
        self.labels = labels
        self.map = [[Neuron(np.random.uniform(0, 1, size=(data.shape[1],))) 
                     for _ in range(m)] for _ in range(n)]
        
    def calc_distance(self, va, vb):
        return math.sqrt(sum((va - vb) **2))
        
    def get_bmu(self, example):
        BMU = None
        min_dist = float('Inf')
        
        for i in range(self.m):
            for j in range(self.n):
                dist = self.calc_distance(self.data[example, :], 
                                          self.map[i][j].weights)
                if dist < min_dist:
                    BMU = (i, j)
                    min_dist = dist
        return BMU
    
    def calc_influence(self, BMU, bmu_radius, cl_rate):
        for i in range(self.m):
            for j in range(self.n):
                dist = self.calc_distance(self.map[BMU[0]][BMU[1]].weights, 
                                          self.map[i][j].weights)
                if dist < bmu_radius:
                    influence = math.exp(-dist/(2*bmu_radius))
                    self.map[BMU[0]][BMU[1]].weights += influence * cl_rate * (
                        self.map[i][j].weights - self.map[BMU[0]][BMU[1]].weights)
        
        
    def train(self, n_cycles, l_rate, labels):
        cl_rate = l_rate
        map_radius = max(self.m, self.n)/2
        time = n_cycles/math.log(map_radius)
        
        for cl in range(n_cycles):
            print('Running cycle %s' % str(cl))
            examples = random.sample(range(self.data.shape[0]), self.data.shape[0])
            for ex in examples:
                BMU = self.get_bmu(ex)
                
                bmu_radius =  map_radius * math.exp(-cl/time)
                self.calc_influence(BMU, bmu_radius, cl_rate)
                cl_rate =  l_rate * math.exp(-cl/n_cycles)
                
                self.map[BMU[0]][BMU[1]].label = labels[ex][0]
                self.map[BMU[0]][BMU[1]].samples.append(ex)
                
    def labels_to_rgb(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.map[i][j].label == -1:
                    self.map[i][j].label = [0, 0, 0]
                elif self.map[i][j].label == 1:
                    self.map[i][j].label = [255, 0, 0]
                elif self.map[i][j].label == 2:
                    self.map[i][j].label = [0, 255, 0]
                elif self.map[i][j].label == 3:
                    self.map[i][j].label = [0, 0, 255]
                   
x, y = Data.get_wine_data()
som = Konehan(x, y, 10, 10)
som.train(1000, 0.01, y)
som.labels_to_rgb()

plt.title('Wine Kohonen')
plt.imshow(np.array([som.map[i][j].label for i in range(len(som.map)) 
        for j in range(len(som.map[0]))]).reshape(len(som.map),len(som.map[0]),3))
plt.show()
