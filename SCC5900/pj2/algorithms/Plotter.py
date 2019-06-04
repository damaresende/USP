'''
Plots the result of clusterization for Prim and Kruskal algorithms.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 30, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''
from matplotlib import pyplot as plt


def display_mst(mst_prim, mst_kruskal, classes_prim, classes_kruskal, datapoints):
    '''
    Plots a 2x2 grid, where on the top plots the MST for Kruskal and Prim algorithms
    is shown, while at the bottom the results for the classification based on DFS is
    displayed.
    
    @param mst_prim: list of nodes and connections for the MST of Prim algorithm [s, v, weight]
    @param mst_kruskal: list of nodes and connections for the MST of Kruskal algorithm [s, v, weight]
    @param classes_prim: list of classes for Prim clusterization
    @param classes_kruskal: list of classes for Kruskal clusterization
    @param datapoints: list of coordinates for each data point in the graph
    @return None
    '''
    x = [p.x for p in datapoints]
    y = [p.y for p in datapoints]
    
    plt.figure(figsize=(12, 10))
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
    
#     plt.show()
    plt.savefig('result.png')
        