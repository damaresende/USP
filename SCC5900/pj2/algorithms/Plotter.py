'''
Created on May 30, 2019

@author: damaresresende
'''
from matplotlib import pyplot as plt


def display_mst(mst_prim, mst_kruskal, datapoints):
    x = [p.x for p in datapoints]
    y = [p.y for p in datapoints]
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(121)
    for k in range(len(mst_prim)):
        plt.plot([x[mst_prim[k][0]], x[mst_prim[k][1]]], 
                 [y[mst_prim[k][0]], y[mst_prim[k][1]]], 
                 '.k-', linewidth=0.5)
    plt.title('Prim')
        
    plt.subplot(122)
    for k in range(len(mst_kruskal)):
        plt.plot([x[mst_kruskal[k][0]], x[mst_kruskal[k][1]]], 
                 [y[mst_kruskal[k][0]], y[mst_kruskal[k][1]]], 
                 '.k-', linewidth=0.5)
    plt.title('Kruskal')         
    plt.show()
        