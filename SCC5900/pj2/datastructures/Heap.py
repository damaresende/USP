'''
Created on May 28, 2019

@author: damaresresende
'''
class MinHeap:
    def __init__(self):
        self.heap = []
        
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        if self.size() == 0:
            return True
        return False
    
    def add(self, key):
        self.heap.append(key)
        self._heapify_up(self.size() - 1)
        
    def poll(self):
        if self.is_empty():
            return
        
        root = self.heap[0]
        
        self.heap[0] = self.heap[-1]
        del self.heap[-1] 
        self._heapify_down(0)
        
        return root
    
    def peek(self):
        if self.is_empty():
            return
        
        return self.heap[0]
    
    def contains(self, v):
        for node in self.heap:
            if node[0] == v:
                return True
        return False
    
    def decrease_key(self, v, weight): 
        i = self._get_pos(v)  
        self.heap[i][1] = weight
        self._heapify_up(i)
      
    def _get_pos(self, v):
        for i, node in enumerate(self.heap):
            if node[0] == v:
                return i
        return -1
    
    def _parent(self, i):
        if i == 0:
            return 0
        return (i - 1) // 2
    
    def _left(self, i):
        return 2 * i + 1
    
    def _right(self, i):
        return 2 * i + 2
    
    def _swap(self, x, y):
        tmp = self.heap[x]
        self.heap[x] = self.heap[y]
        self.heap[y] = tmp
        
    def _heapify_down(self, i):
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
        parent = self._parent(i)
        
        if i > 0 and self.heap[parent][1] > self.heap[i][1]:
            self._swap(i, parent)
            self._heapify_up(parent)
