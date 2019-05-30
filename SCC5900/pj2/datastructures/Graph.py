'''
Created on May 28, 2019

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
        self.npoints = len(self.datapoints)
        self.graph = self._build_graph()
        
    def _get_data_points(self):
        datapoints = []
        root_path = os.path.join(os.getcwd().split('pj2')[0], 'pj2')
        
        with open(os.path.join(os.path.join(root_path, 'data'), 'data.txt')) as f:
            for line in f.readlines():
                x, y = line.split('\t')
                datapoints.append(Point(float(x), float(y)))
                
        return datapoints
        
    def _build_graph(self):
        graph = [None] * self.npoints * self.npoints
        
        for s in range(self.npoints):
            for v in range(self.npoints):
                if s == v:
                    graph[s * self.npoints + v] = [s, v, float('inf')]
                else:
                    weight = self.datapoints[s].calc_eucledean_distance(self.datapoints[v])
                    graph[s * self.npoints + v] = [s, v, weight]
            
        return graph
    
    def get_weight(self, s, v):
        return self.graph[s * self.npoints + v][2]
    
    def display_graph(self):
        x = [p.x for p in self.datapoints]
        y = [p.y for p in self.datapoints]
        
        for s in range(self.npoints):
            for v in range(self.npoints):
                if self.graph[s * self.npoints + v][2] < float('inf'):
                    plt.plot([x[s], x[v]], [y[s], y[v]], '.k-', linewidth=0.5)
                
        plt.scatter(x, y, s=[20 for _ in range(len(x))])
        plt.show()