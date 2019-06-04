'''
Reads the points defined in data/data.txt and builds a graph data
structure with them. The graph is fully connected. Each node is like 
[s, v, w] were s is the source, v the destination and w the weight.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 28, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''
import os
from math import sqrt


class Point:
    def __init__(self, x, y):
        '''
        Initializes coordinates
        
        @param x: x coordinate
        @param y: y coordinate
        @return None
        '''
        self.x = x
        self.y = y
    
    def calc_euclidean_distance(self, datapoint):
        '''
        Calculates the Euclidean distance between the current vertex and
        the one defined in datapoint object
        
        @param datapoint: Point with vertex coordinates
        @return float with Euclidean distance value
        '''
        dx = self.x - datapoint.x
        dy = self.y - datapoint.y
        return sqrt(dx * dx + dy * dy)

    
class Graph:
    def __init__(self):
        '''
        Initializes the graph by reading the data points in data/data.txt
        and building the graph list of nodes
        '''
        self.datapoints = self._get_data_points()
        self.npoints = len(self.datapoints)
        self.graph = self._build_graph()
        
    def _get_data_points(self):
        '''
        Reads the datapoins in data/data.txt and stores them in a list of
        Points that defines each point coordinates
        
        @return list of Points with data points
        '''
        datapoints = []
        root_path = os.path.join(os.getcwd().split('pj2')[0], 'pj2')
        
        with open(os.path.join(os.path.join(root_path, 'data'), 'data.txt')) as f:
            for line in f.readlines():
                x, y = line.split('\t')
                datapoints.append(Point(float(x), float(y)))
                
        return datapoints
        
    def _build_graph(self):
        '''
        Builds the graph with a list of nodes. Each node is like [s, v, w] were s is 
        the source, v the destination and w the weight.
        
        @return list of nodes in the graph. The list has V * V nodes, where V is the number
        of points in the graph
        '''
        graph = [None] * self.npoints * self.npoints
        
        for s in range(self.npoints):
            for v in range(self.npoints):
                if s == v:
                    graph[s * self.npoints + v] = [s, v, float('inf')]
                else:
                    weight = self.datapoints[s].calc_euclidean_distance(self.datapoints[v])
                    graph[s * self.npoints + v] = [s, v, weight]
            
        return graph
    
    def get_weight(self, s, v):
        '''
        Retrieves the weight of a node s connected to v
        '''
        return self.graph[s * self.npoints + v][2]