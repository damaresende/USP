'''
Created on May 26, 2019

@author: damaresresende
'''
from matplotlib import pyplot as plt

from SCC5900.pj2.datastructures.Graph import Graph


class Kruskal:
    def build_mst(self, graph, npoints):
        result = []
        
        i = 0
        e = 0
        
        graph = sorted(graph, key=lambda item: item[2])
        parent = []
        rank = []
        
        for node in range(npoints):
            parent.append(node)
            rank.append(0)
        
        while e < npoints - 1:
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
            
    def display_mst(self, mst, datapoints):
        x = [p.x for p in datapoints]
        y = [p.y for p in datapoints]
        
        for k in range(len(mst)):
            plt.plot([x[mst[k][0]], x[mst[k][1]]], [y[mst[k][0]], y[mst[k][1]]], '.k-', linewidth=0.5)
                
        plt.scatter(x, y, s=[10 for _ in range(len(x))])
        plt.show()
        
g = Graph()
# g.display_graph()
k = Kruskal()
mst = k.build_mst(g.graph, g.npoints)
k.display_mst(mst, g.datapoints)