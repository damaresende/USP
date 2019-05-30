'''
Created on May 25, 2019

@author: damaresresende
'''
from SCC5900.pj2.datastructures.Heap import MinHeap

 
class Prim:
    def __init__(self, nclusters):
        self.nclusters = nclusters
    
    def build_mst(self, graph, npoints): 
        
        heap = MinHeap(npoints)
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
                
                if heap.contains(v) and graph[u * npoints+ x][2] < key[v]: 
                    key[v] = graph[u * npoints + x][2]
                    result[v - 1] = [u, v, graph[u * npoints + x][2]]
  
                    heap.decrease_key(v, key[v]) 
  
        self._remove_large_edges(result)
        return result
    
    def _remove_large_edges(self, mst):
        for _ in range(self.nclusters-1):
            idx = -1
            max_ = -float('inf')
         
            for i, e in enumerate(mst):
                if e[2] > max_:
                    max_ = e[2]
                    idx = i
            del mst[idx]
