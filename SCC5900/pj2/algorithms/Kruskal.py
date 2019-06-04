'''
Uses union find data structure with ranking and path reduction to build a MST
based on Kruskal's algorithm. The algorithm runs V - 1 - K - 1 times, where V is
the number of vertices and K the number of clusters to be found.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 26, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''
class Kruskal:
    def __init__(self, nclusters):
        '''
        Initializes the number of clusters
        
        @param nclusters: number of clusters
        @return None
        '''
        self.nclusters = nclusters
        
    def build_mst(self, graph, npoints):
        '''
        Builds the MST for Kruskal's algorithm. It runs until all clusters are found.
        
        @param graph: List of nodes in the graph. Each node is like [s, v, w] were s is the source, 
        v the destination and w the weight
        @param npoints: number of points in the graph
        @return list of nodes in Minimum Spanning Tree. Each node is like [s, v, w]
        were s is the source, v the destination and w the weight
        '''
        i = 0
        e = 0
        
        rank = [] # stores the rank of each set
        result = [] # stores the MST formed
        parent = [] # stores the parent of each set
        graph = sorted(graph, key=lambda item: item[2]) # sorts the graph so the smallest edges
        # are picked first. This guarantees that the K largest edges will not be considered
        
        # initializes rank and parent arrays
        for node in range(npoints):
            parent.append(node)
            rank.append(0)
        
        # while I do not reach the number of vertices minus the root minus the number of clusters minus one
        while e < (npoints - 1) - (self.nclusters - 1):
            u, v, w = graph[i] # gets source, destination and weight of the graph node
            i = i + 1
            x = self.find(parent, u) # finds the node parent
            y = self.find(parent, v) # finds the node parent
            
            # if the inclusion of the edge does not form a circle
            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y) # add the set to the MST
                
        return result
    
    def find(self, parent, i):
        '''
        Finds the set of an element i by using the path compression method
        
        @param parent: parent set id
        @param i: node id
        @return the set connected to i
        '''
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
    
    def union(self, parent, rank, x, y):
        '''
        Performs the union of two sets by using the rank of them
        
        @param parent: parent set
        @param rank: set rank
        @param x: set x
        @param y: set y
        @return None
        '''
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        
        # Attach smaller rank tree under root of high rank tree
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            # If ranks are same, then make one as root and increment its rank by one
            parent[yroot] = xroot
            rank[xroot] += 1