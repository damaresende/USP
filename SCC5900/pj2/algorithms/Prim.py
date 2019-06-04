'''
Uses heap data structure to build a MST based on Prim's algorithm. 
The algorithm runs until there is a node not added to the graph and later
removes the K - 1 edges to form K of clusters.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 25, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''
from SCC5900.pj2.datastructures.Heap import MinHeap

 
class Prim:
    def __init__(self, nclusters):
        '''
        Initializes the number of clusters
        
        @param nclusters: number of clusters
        @return None
        '''
        self.nclusters = nclusters
    
    def build_mst(self, graph, npoints): 
        '''
        Builds the MST for Prim's algorithm. It runs until there is no node left to
        add to the graph and later removes the K - 1 largest edges
        
        @param graph: List of nodes in the graph. Each node is like [s, v, w] were s is the source, 
        v the destination and w the weight
        @param npoints: number of points in the graph
        @return list of nodes in Minimum Spanning Tree. Each node is like [s, v, w]
        were s is the source, v the destination and w the weight
        '''
        heap = MinHeap(npoints) # initializes heap
        key = [float('inf')] * npoints    
        result = [[-1, -1, float('inf')]] * (npoints - 1)  # initializes MST

        # adds all keys to the heap with infinity weight
        for v in range(npoints):
            heap.add([v, key[v]])
  
        key[0] = 0 # defines the root as node 0
        heap.decrease_key(0, key[0]) # weight of node 0 in the heap is 0
  
        while heap.is_empty() == False: # while I have nodes to be added

            u = heap.poll()[0] # removes the node with minimum weight
            
            # checks in all points if its weight is less than the one already set
            for x in range(npoints): 
                v = graph[u * npoints + x][1]
                
                # if my heap has that vertex and its weight is less than what set before
                # I should update the heap
                if heap.contains(v) and graph[u * npoints+ x][2] < key[v]: 
                    key[v] = graph[u * npoints + x][2] # updates key weights
                    result[v - 1] = [u, v, graph[u * npoints + x][2]] # sets the node in the MST
  
                    heap.decrease_key(v, key[v]) # updates heap
  
        self._remove_large_edges(result) # removes K - 1 largest edges
        return result
    
    def _remove_large_edges(self, mst):
        '''
        Removes the K - 1 largest edges in the MST
        @param mst: list of nodes in Minimum Spanning Tree. Each node is like [s, v, w] 
        were s is the source, v the destination and w the weight
        @return None
        '''
        # deletes largest edge to form K clusters
        for _ in range(self.nclusters-1):
            idx = -1
            max_ = -float('inf')
         
            # iterates through all nodes, finds the largest edge and deletes it
            for i, e in enumerate(mst):
                if e[2] > max_:
                    max_ = e[2]
                    idx = i
            del mst[idx]
