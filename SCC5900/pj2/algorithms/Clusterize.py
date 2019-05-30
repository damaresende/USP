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

start_time = time.time()
  
p = Prim(nclusters) 
mst_prim = p.build_mst(g.graph, g.npoints)
print("Prim running time: %s seconds" % (time.time() - start_time))

start_time = time.time()
   
k = Kruskal(nclusters)
mst_kruskal = k.build_mst(g.graph, g.npoints)
print("Kruskal running time: %s seconds" % (time.time() - start_time))

Plotter.display_mst(mst_prim, mst_kruskal, g.datapoints)