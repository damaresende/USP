'''
Created on May 30, 2019

@author: damaresresende
'''
import time

from SCC5900.pj2.algorithms.Prim import Prim
from SCC5900.pj2.datastructures.Graph import Graph
from SCC5900.pj2.algorithms.Kruskal import Kruskal

g = Graph()
start_time = time.time()

p = Prim() 
mst = p.build_mst(g.graph, g.npoints)
print("Prim running time: %s seconds" % (time.time() - start_time))

start_time = time.time()

k = Kruskal()
mst = k.build_mst(g.graph, g.npoints)
print("Kruskal running time: %s seconds" % (time.time() - start_time))


# g.display_graph()
# k.display_mst(mst, g.datapoints)
# p.display_mst(mst, g.datapoints)