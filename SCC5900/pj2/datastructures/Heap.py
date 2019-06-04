'''
Minimum Heap data structure to be used by Prim's algorithm 

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 28, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Project of Algorithms Class (SCC5000)
'''
class MinHeap:
    def __init__(self, npoints):
        '''
        Initializes heap
        
        @param npoints: number of points in the graph
        '''
        self.heap = []
        # auxiliary structure to store the ID of vertices that are in the heap
        # this makes the method contains O(n)
        self.vertices = {v: False for v in range(npoints)}
        
    def size(self):
        '''
        Retrieves the size of the heap
        
        @return integer with the size of the heap
        '''
        return len(self.heap)
    
    def is_empty(self):
        '''
        Checks if heap is empty
        
        @return True if heap is empty, False otherwise
        '''
        if self.size() == 0:
            return True
        return False
    
    def add(self, key):
        '''
        Adds an element to the heap and then moves it to the top in
        case its parent is greater than it
        
        @param key: [x w] where x is the node and w the weight
        '''
        self.heap.append(key)
        self.vertices[key[0]] = True # sets the flag for that
                                    # vertex to true to indicate it is in the heap
        self._heapify_up(self.size() - 1) # updates heap
        
    def poll(self):
        '''
        Removes the root of the heap and adjusts it so root will always
        store the minimum value
        
        @return [x w] where x is the node and w the weight
        '''
        if self.is_empty():
            return
        
        root = self.heap[0]
        
        self.heap[0] = self.heap[-1]
        del self.heap[-1]
        self._heapify_down(0) # updates heap
        self.vertices[root[0]] = False # sets the flag for that
                                    # vertex to false to indicate it is not in the heap
        
        return root
    
    def peek(self):
        '''
        Retrieves the minimum value in the heap (root value) but does not remove it
        
        @return [x w] where x is the node and w the weight
        '''
        if self.is_empty():
            return
        
        return self.heap[0]
    
    def contains(self, v):
        '''
        Checks if heap contains vertex
        
        @return True if heap has vertex, false otherwise
        '''
        return self.vertices[v]
    
    def decrease_key(self, v, weight): 
        '''
        Updates the weight value of the heap node v and then adjusts the heap
        to make sure that no parent is greater than its children
        
        @param v: vertex ID
        @param weight: vertex weight
        @return None
        '''
        i = self._get_pos(v)  
        self.heap[i][1] = weight
        self._heapify_up(i)
      
    def _get_pos(self, v):
        '''
        Gets the index position of vertex v in the heap
        
        @param v: vertex ID
        '''
        for i, node in enumerate(self.heap):
            if node[0] == v:
                return i
        return -1
    
    def _parent(self, i):
        '''
        Gets parent node
        
        @param i: node ID
        @return parent node ID
        '''
        if i == 0:
            return 0
        return (i - 1) // 2
    
    def _left(self, i):
        '''
        Gets left node
        
        @param i: node ID
        @return left node ID
        '''
        return 2 * i + 1
    
    def _right(self, i):
        '''
        Gets right node
        
        @param i: node ID
        @return right node ID
        '''
        return 2 * i + 2
    
    def _swap(self, x, y):
        '''
        Swaps two nodes
        
        @param x: node a
        @param y: node b
        @return None
        '''
        tmp = self.heap[x]
        self.heap[x] = self.heap[y]
        self.heap[y] = tmp
        
    def _heapify_down(self, i):
        '''
        Swaps the nodes in the tree until the node that is large goes to the bottom
        of the heap and the parent of it, if there is any, is greater than it
        
        @param i: node ID
        @return None
        '''
        left = self._left(i)
        right = self._right(i)
        
        smallest = i
        if left < self.size() and self.heap[left][1] < self.heap[i][1]:
            smallest = left
        
        if right < self.size() and self.heap[right][1] < self.heap[smallest][1]:
            smallest = right
            
        if smallest != i:
            self._swap(i, smallest)
            self._heapify_down(smallest)
            
    def _heapify_up(self, i):
        '''
        Swaps the nodes in the tree until the node that is small goes to the top
        of the heap and the parent of it, if there is any, is smaller than it
        
        @param i: node ID
        @return None
        '''
        parent = self._parent(i)
        
        if i > 0 and self.heap[parent][1] > self.heap[i][1]:
            self._swap(i, parent)
            self._heapify_up(parent)
