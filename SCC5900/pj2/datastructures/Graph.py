'''
Created on May 25, 2019

@author: damaresresende
'''
import os
from math import sqrt
from matplotlib import pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def calc_eucledean_distance(self, datapoint):
        dx = self.x - datapoint.x
        dy = self.y - datapoint.y
        return sqrt(dx * dx + dy * dy)
        

class Graph:
    def __init__(self):
        self.datapoints = self._get_data_points()
        self.adjancymatrix = self._build_adjancy_matrix()
        
    def _get_data_points(self):
        datapoints = []
        root_path = os.path.join(os.getcwd().split('pj2')[0], 'pj2')
        
        with open(os.path.join(os.path.join(root_path, 'data'), 'data_bkp.txt')) as f:
            for line in f.readlines():
                x, y = line.split('\t')
                datapoints.append(Point(float(x), float(y)))
                
        return datapoints
        
    def _build_adjancy_matrix(self):
        npoints = len(self.datapoints)
        adjancymatrix = [[0 for _ in range(npoints)] for _ in range(npoints)]
        
        for i in range(npoints):
            for j in range(npoints):
                if i == j:
                    adjancymatrix[i][j] = 0
                else:
                    adjancymatrix[i][j] = self.datapoints[i].calc_eucledean_distance(self.datapoints[j])
            
        return adjancymatrix
    
    def display_graph(self, graph):
        x = [p.x for p in self.datapoints]
        y = [p.y for p in self.datapoints]
        
        for w in range(len(self.datapoints)):
            for k in range(1, len(self.datapoints)):
                if graph[w][k] > 0:
                    plt.plot([x[w], x[k]], [y[w], y[k]], '.k-', linewidth=0.5)
                
        plt.scatter(x[0], y[0], s=[10 for _ in range(len(x))])
        plt.grid(True)
        plt.show()
