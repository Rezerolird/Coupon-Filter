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

    def GetVector(self, vec):
        location = np.floor((np.dot(self.a, vec) + self.b) / self.w)
        return location

    def GetIndex(self, index):
        idx = np.dot(self.lsh_r, index)
        return int(idx)

class MLBF:
    def __init__(self, num_sets:int, hash_k:int, hash_l:int, dim:int, max_ele:int, w:float):
        print('Start initializing...')
        self.hash_k = hash_k
        self.hash_l = hash_l
        self.hash_num = self.hash_k*self.hash_l
        self.num_sets = num_sets
        self.dim = dim
        self.max_ele = max_ele
        self.w = w
        self.hash_func = [ HashFunc(1, self.dim, self.w) for _ in range(self.hash_num) ]

        self.single_capacity = int(np.ceil(self.max_ele / self.num_sets/self.hash_num))
        self.bf = [ [bitarray.bitarray([False]) * self.single_capacity for _ in range(self.hash_num)] for _ in range(self.num_sets) ]
        print('Initialization finished!')
        self.visited = [ [0] * self.single_capacity for _ in range(self.num_sets) ]
        self.count = 0
        self.flag = 0
        self.insert_time = 0
    def Insert(self, SetID:int, vector):
        t = time.time()
        for i in range(self.hash_num):
            location_vec = self.hash_func[i].GetVector(vector) 
            location = self.hash_func[i].GetIndex(location_vec)   % self.single_capacity
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
        t0 = time.time()
        locations = []
        for i in range(self.hash_l):
            for k in range(self.hash_k):
                location_vec = self.hash_func[i*self.hash_k+k].GetVector(vector)
                location = self.hash_func[i*self.hash_k+k].GetIndex(location_vec)  % self.single_capacity
                locations.append(location)
        
        results = [];

        for j in range(self.num_sets):
            for i in range(self.hash_l):
                fg = 1
                for k in range(self.hash_k):
                    location = locations[i*self.hash_k+k]
                    if self.bf[j][i*self.hash_k+k][location] == 0:
                        fg = 0
                        break
                if fg == 1:
                    results.append(j)
                    break

        t1 = time.time()
        return results, t1-t0

