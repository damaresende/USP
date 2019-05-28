'''
Created on May 25, 2019

@author: damaresresende
'''
from matplotlib import pyplot as plt

from SCC5900.pj2.datastructures.Graph import Graph


class Prim:
    def __init__(self, npoints):
        self.key = [float('inf')] * npoints
        self.mstSet = [False] * npoints
        self.mst = [0] * npoints
        
        self.mst[0] = -1
        self.key[0] = 0 
        self.npoints = npoints
        
    def build_mst(self, graph):
        for _ in range(self.npoints):
            u = self.minKey()
            self.mstSet[u] = True
            
            for v in range(self.npoints):
                if graph[u][v] > 0 and not self.mstSet[v] and self.key[v] > graph[u][v]: 
                    self.key[v] = graph[u][v] 
                    self.mst[v] = u
    
    def minKey(self): 
        min_ = float('inf') 
  
        for v in range(self.npoints): 
            if self.key[v] < min_ and not self.mstSet[v]: 
                min_ = self.key[v] 
                min_index = v 
  
        return min_index
    
    def remove_max_vertices(self, k):
        for _ in range(k):
            max_ = -1
            max_idx = 0
            
            for i, v in enumerate(self.key):
                if v > max_:
                    max_ = v
                    max_idx = i
            
            self.key[max_idx] = -1
            self.mst[max_idx] = -2
            
            
    def display_mst(self, mst, datapoints):
        x = [p.x for p in datapoints]
        y = [p.y for p in datapoints]
        
        for k in range(len(mst)):
            if mst[k] >= 0:
                plt.plot([x[k], x[mst[k]]], [y[k], y[mst[k]]], '.k-', linewidth=0.5)
                
        plt.scatter(x, y, s=[10 for _ in range(len(x))])
        plt.show()
        
    
g = Graph()
p = Prim(len(g.datapoints))


# g.display_graph(g.adjancymatrix)

# for row in g.adjancymatrix:
#     print(row)
    
p.build_mst(g.adjancymatrix)
# p.remove_max_vertices(4)

print(p.mst)
print(p.key)

p.display_mst(p.mst, g.datapoints)