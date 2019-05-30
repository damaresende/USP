'''
Created on May 30, 2019

@author: damaresresende
'''
import time

from SCC5900.pj2.algorithms.Prim import Prim
from SCC5900.pj2.datastructures.Graph import Graph
from SCC5900.pj2.algorithms.Kruskal import Kruskal
from SCC5900.pj2.algorithms import Plotter


nclusters = 5
g = Graph()

# start_time = time.time()
#   
# p = Prim(nclusters) 
# mst_prim = p.build_mst(g.graph, g.npoints)
# print("Prim running time: %s seconds" % (time.time() - start_time))

start_time = time.time()
   
k = Kruskal(nclusters)
mst_kruskal = k.build_mst(g.graph, g.npoints)
print("Kruskal running time: %s seconds" % (time.time() - start_time))

# Plotter.display_mst(mst_prim, mst_kruskal, g.datapoints)

mst = {v: [] for v in range(g.npoints)}
for node in mst_kruskal:
    mst[node[0]].append(node[1])
    mst[node[1]].append(node[0])
    
def DFS(graph, datapoints): 
    cluster = []
    vertices = set()
    
    def _traverse(v, visited):  
        visited[v] = True
        vertices.add(v)
        
        for n in graph[v]: 
            if visited[n] == False: 
                vertices.add(n)
                cluster.append(datapoints[v])
                _traverse(n, visited) 

    visited = {key: False for key in graph} 

    root = list(mst.keys())[0]
    cluster.append(datapoints[root])
    _traverse(root, visited)
        
    for x in vertices:
        graph.pop(x)
    
    return cluster    

cluster = DFS(mst, g.datapoints)
cluster = DFS(mst, g.datapoints)

from matplotlib import pyplot as plt

x = [p.x for p in cluster]
y = [p.y for p in cluster]

plt.scatter(x, y, s=[40 for _ in range(len(x))])
plt.show()
