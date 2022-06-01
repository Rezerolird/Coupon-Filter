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

class LSBF:
    def __init__(self, num_sets:int, hash_k:int, hash_l:int, dim:int, max_ele:int, w:float):
        print('Start initializing...')
        self.hash_k = hash_k
        self.hash_l = hash_l
        self.num_sets = num_sets
        self.dim = dim
        self.max_ele = max_ele
        self.w = w
        self.hash_func = [ HashFunc(self.hash_k, self.dim, self.w) for _ in range(self.hash_l) ]


        self.single_capacity = int(np.ceil(max_ele / self.num_sets/self.hash_l))

        self.bf = [ [bitarray.bitarray([False]) * self.single_capacity for _ in range(self.hash_l)] for _ in range(self.num_sets) ]

        # self.bf = [ bitarray.bitarray([False]) * self.capacity for _ in range(self.num_sets) ]
        self.visited = [ [0] * self.single_capacity for _ in range(self.num_sets) ]
        self.count = 0
        self.flag = 0
        self.insert_time = 0
        print('Initialization finished!')


    def Insert(self, SetID:int, vector):
        t = time.time()
        for i in range(self.hash_l):
            location_vec = self.hash_func[i].GetVector(vector) 
            location = self.hash_func[i].GetIndex(location_vec) % self.single_capacity
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
            
        results = []
        t0 = time.time()

        locations = []
        for i in range(self.hash_l):
            location_vec = self.hash_func[i].GetVector(vector) 
            location = self.hash_func[i].GetIndex(location_vec) % self.single_capacity
            locations.append(location)

        for j in range(self.num_sets):
            flag = 1
            for i in range(self.hash_l):                
                location = locations[i]
                if(self.bf[j][i][location]):
                    continue
                else:
                    for k in range(self.hash_k):
                        location_vec[k] -= 1
                        new_loc = self.hash_func[i].GetIndex(location_vec) % self.single_capacity
                        if(self.bf[j][i][new_loc]):
                            break
                        location_vec[k] += 2
                        new_loc = self.hash_func[i].GetIndex(location_vec) % self.single_capacity
                        if(self.bf[j][i][new_loc]):
                            break
                        location_vec[k] -= 1
                        if k == self.hash_k-1:
                            flag = 0
                            break
                if flag == 0:
                    break
            if flag:
                results.append(j)
                
        t1 = time.time()
        return results, t1-t0
        


