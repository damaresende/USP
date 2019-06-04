'''
Runs clusterization for Prim and Kruskal algorithms based on the graph described in
data/data.txt file. The graph is considered to be fully connected at first and then
each algorithm builds a MST to minimize the number of edges. K is an input value to
determine the number of clusters. The K-1 largest edges in each MST are disconsidered 

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 30, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''
import sys
import time

from SCC5900.pj2.algorithms.Prim import Prim
from SCC5900.pj2.datastructures.Graph import Graph
from SCC5900.pj2.algorithms.Kruskal import Kruskal
from SCC5900.pj2.algorithms import Plotter
    

def DFS(graph, datapoints, klass):
    '''
    Depth First Search algorithm to find the K clusters in the
    disconnected MST
    
    @param graph: dictionary of vertices in the graph, where the key is the source
    vertex and the value is a list of connected vertices
    @param datapoints: list of data points classes
    @param klass: integer with cluster class
    @return None
    '''
    vertices = set() # set of vertices that belong to the cluster
    
    def _traverse(v):
        '''
        Auxiliary method to be used in recursion that traverses the graph
        
        @param v: node being visited
        @return None
        '''
        visited[v] = True # marks node as visited
        vertices.add(v) # adds it to the list of cluster nodes
        
        # checks all connections for node v
        for n in graph[v]: 
            if visited[n] == False: 
                # if the node was not already visited, add it to the set of
                # cluster nodes, determine its class and traverse the other
                # nodes that may be connected to it
                vertices.add(n)
                datapoints[v] = klass
                _traverse(n) 

    # dictionary that marks the visited nodes
    visited = {key: False for key in graph} 

    root = list(graph.keys())[0] # root node
    datapoints[root] = klass # root node class
    _traverse(root) # start traversal
        
    # remove the vertices in the cluster from the graph so other clusters can be found
    for x in vertices:
        graph.pop(x)

def classify(mst, nclusters, g):
    '''
    Finds K clusters in the disconnected MST and classifies them
    
    @param mst: list of nodes in Minimum Spanning Tree. Each node is like [s, v, w]
    were s is the source, v the destination and w the weight
    @param nclusters: number of clusters to be created
    @param g: graph data structure
    @return array of classes for each graph vertex
    '''
    
    # transforms the list of nodes in a dictionary of connections for the DFS algorithm
    graph = {v: [] for v in range(g.npoints)}
    for node in mst:
        graph[node[0]].append(node[1]) # edge that goes
        graph[node[1]].append(node[0]) # edge that comes back
        
    classes = [-1] * len(g.datapoints) # initializes all classes as -1
    for k in range(nclusters): # calls DFS to find each cluster
        DFS(graph, classes, k)
        
    return classes

def run(nclusters):
    '''
    Runs application
    
    @param nclusters: number of clusters to form
    @return None
    '''
    g = Graph()
    
    # calls Prim's algorithm and builds its MST
    start_time = time.time()
       
    p = Prim(nclusters) 
    mst_prim = p.build_mst(g.graph, g.npoints)
    print("Prim running time: %s seconds" % (time.time() - start_time))
    
    # calls Kruskal's algorithm and builds its MST
    start_time = time.time()
       
    k = Kruskal(nclusters)
    mst_kruskal = k.build_mst(g.graph, g.npoints)
    print("Kruskal running time: %s seconds" % (time.time() - start_time))
    
    # classifies each MST formed
    classes_p = classify(mst_prim, nclusters, g)
    classes_k = classify(mst_kruskal, nclusters, g)
    
    # plots results
    Plotter.display_mst(mst_prim, mst_kruskal, classes_p, classes_k, g.datapoints)
    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(int(sys.argv[1]))
    else:
        run(6)
    