'''
Created on May 30, 2019

@author: damaresresende
'''
from matplotlib import pyplot as plt


def display_mst(mst_prim, mst_kruskal, classes_prim, classes_kruskal, datapoints):
    x = [p.x for p in datapoints]
    y = [p.y for p in datapoints]
    
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    
    plt.subplot(221)
    for k in range(len(mst_prim)):
        plt.plot([x[mst_prim[k][0]], x[mst_prim[k][1]]], 
                 [y[mst_prim[k][0]], y[mst_prim[k][1]]], 
                 '.k-', linewidth=0.5)
    plt.title('Prim MST')
    plt.axis('off')
        
    plt.subplot(222)
    for k in range(len(mst_kruskal)):
        plt.plot([x[mst_kruskal[k][0]], x[mst_kruskal[k][1]]], 
                 [y[mst_kruskal[k][0]], y[mst_kruskal[k][1]]], 
                 '.k-', linewidth=0.5)
    plt.title('Kruskal MST')
    plt.axis('off')

    plt.subplot(223)
    plt.scatter(x, y, c=classes_prim, s=[10 for _ in range(len(x))])
    plt.title('Prim Classification')
    plt.axis('off')
    
    plt.subplot(224)
    plt.scatter(x, y, c=classes_kruskal, s=[10 for _ in range(len(x))])
    plt.title('Kruskal Classification')
    plt.axis('off')
    
    plt.show()
        