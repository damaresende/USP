'''
Created on May 25, 2019

@author: damaresresende
'''
from matplotlib import pyplot as plt

from SCC5900.pj2.datastructures.Graph import Graph
from SCC5900.pj2.datastructures.Heap import MinHeap

 
class Prim:
    def build_mst(self, graph, npoints): 
        
        heap = MinHeap()
        key = [float('inf')] * npoints    
        result = [[-1, -1, float('inf')]] * (npoints - 1)  

        for v in range(npoints):
            heap.add([v, key[v]])
  
        key[0] = 0
        heap.decrease_key(0, key[0]) 
  
        while heap.is_empty() == False: 

            u = heap.poll()[0] 
            
            for x in range(npoints): 
                v = graph[u * npoints + x][1]
                
                if heap.contains (v) and graph[u * npoints+ x][2] < key[v]: 
                    key[v] = graph[u * npoints + x][2]
                    result[v - 1] = [u, v, graph[u * npoints + x][2]]
  
                    heap.decrease_key(v, key[v]) 
  
        return result
  
    def display_mst(self, mst, datapoints):
        x = [p.x for p in datapoints]
        y = [p.y for p in datapoints]
        
        for k in range(len(mst)):
            plt.plot([x[mst[k][0]], x[mst[k][1]]], [y[mst[k][0]], y[mst[k][1]]], '.k-', linewidth=0.5)
                
        plt.scatter(x, y, s=[10 for _ in range(len(x))])
        plt.show()


g = Graph()
# g.display_graph()
p = Prim() 
mst = p.build_mst(g.graph, g.npoints) 
p.display_mst(mst, g.datapoints)