'''
Created on May 30, 2019

@author: damaresresende
'''
import time

from SCC5900.pj2.algorithms.Prim import Prim
from SCC5900.pj2.datastructures.Graph import Graph
from SCC5900.pj2.algorithms.Kruskal import Kruskal
from SCC5900.pj2.algorithms import Plotter
    
def DFS(graph, datapoints, klass): 
    vertices = set()
    
    def _traverse(v, visited):  
        visited[v] = True
        vertices.add(v)
        
        for n in graph[v]: 
            if visited[n] == False: 
                vertices.add(n)
                datapoints[v] = klass
                _traverse(n, visited) 

    visited = {key: False for key in graph} 

    root = list(graph.keys())[0]
    datapoints[root] = klass
    _traverse(root, visited)
        
    for x in vertices:
        graph.pop(x)

def classify(mst, nclusters):
    graph = {v: [] for v in range(g.npoints)}
    for node in mst:
        graph[node[0]].append(node[1])
        graph[node[1]].append(node[0])
        
    classes = [-1] * len(g.datapoints)
    for k in range(nclusters):
        DFS(graph, classes, k)
        
    return classes

nclusters = 6
g = Graph()

start_time = time.time()
   
p = Prim(nclusters) 
mst_prim = p.build_mst(g.graph, g.npoints)
print("Prim running time: %s seconds" % (time.time() - start_time))

start_time = time.time()
   
k = Kruskal(nclusters)
mst_kruskal = k.build_mst(g.graph, g.npoints)
print("Kruskal running time: %s seconds" % (time.time() - start_time))

classes_p = classify(mst_prim, nclusters)
classes_k = classify(mst_kruskal, nclusters)

Plotter.display_mst(mst_prim, mst_kruskal, classes_p, classes_k, g.datapoints)
