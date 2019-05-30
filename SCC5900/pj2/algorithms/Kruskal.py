'''
Created on May 26, 2019

@author: damaresresende
'''
class Kruskal:
    def __init__(self, nclusters):
        self.nclusters = nclusters
        
    def build_mst(self, graph, npoints):
        i = 0
        e = 0
        
        rank = []
        result = []
        parent = []
        graph = sorted(graph, key=lambda item: item[2])
        
        for node in range(npoints):
            parent.append(node)
            rank.append(0)
        
        while e < (npoints - 1) - (self.nclusters - 1):
            u, v, w = graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            
            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
                
        return result
    
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
    
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1