import mmh3
import bitarray
import time
import numpy as np
import random
from scipy.special import ndtr
import math



class HashFunc:
    def __init__(self, k:int, dim:int, w:float):
        self.k = k
        self.dim = dim
        self.w = w
        self.a = np.random.standard_normal(size=(self.k, self.dim))
        self.b = np.random.uniform(0, self.w, size=(self.k))
        self.lsh_r = np.random.randint(536870912, size=(1, self.k))
        self.seed = np.random.randint(536870912)
        
    def GetIndex(self, vec):

        location = np.floor((np.dot(self.a, vec) + self.b) / self.w)
        idx = np.dot(self.lsh_r, location)
        return mmh3.hash(str(int(idx)), self.seed)

class DSFilter:
    def __init__(self, num_sets: int, hash_num: int, dim: int, max_ele:int, w:float, threshold: int):
        print('Start initializing ...')
        self.hash_num = hash_num
        self.num_sets = num_sets
        self.dim = dim
        self.max_ele = max_ele
        self.w = w
        self.hash_func = [ HashFunc(1, self.dim, self.w) for _ in range(self.hash_num)]

        self.single_capacity = int(np.ceil(max_ele / self.num_sets/self.hash_num))
        self.bf = [ [bitarray.bitarray([False]) * self.single_capacity for _ in range(self.hash_num)] for _ in range(self.num_sets) ]

        self.t = threshold
        self.visited = [ [0] * self.single_capacity for _ in range(self.num_sets) ]
        self.count = 0
        self.flag = 0
        self.insert_time = 0
        print('Initialization finished !')
        
    def Insert(self, SetID:int, vector):
        t = time.time()
        for i in range(self.hash_num):
            location = self.hash_func[i].GetIndex(vector) % self.single_capacity
            self.bf[SetID][i][location] = True
            if i == 0 and self.visited[SetID][location] == 0:
                self.visited[SetID][location] = 1
                self.count += 1
        self.insert_time += time.time()-t

    def Query(self, vector):
        if self.flag == 0:
            print(self.count)
            print(self.insert_time)
            print("")
            self.flag = 1
        results = []; locations = []
        t0 = time.time()
        for i in range(self.hash_num):
            location = self.hash_func[i].GetIndex(vector) % self.single_capacity
            locations.append(location)
        t1 = time.time()


        for i in range(self.num_sets):
            count = 0
            for j in range(self.hash_num):
                if self.bf[i][j][locations[j]]:
                    count += 1
                if count == self.t:
                    results.append(i)
                    break
        t2 = time.time()
        
        return results, t2-t0
