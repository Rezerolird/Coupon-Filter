from re import S
import mmh3
import bitarray
import time
import numpy as np
import random
from scipy.special import ndtr
import math
import ffht

class HashFunc:
    def __init__(self, hash_num:int, dim:int, w:float):
        self.dim = dim
        self.rot_num = 3
        self.N = hash_num
        self.w = w

        self.hadamard_order = int(np.ceil(np.log2(self.dim))) 
        self.piece_length = 2**self.hadamard_order 
        self.piece_number = int(np.ceil(self.N/self.piece_length)) 
        self.real_N = self.piece_number * self.piece_length 
        self.flip = np.sign(np.random.normal(size=(self.rot_num, self.real_N)))
        self.lsh_r = np.random.randint(1, 536870912)
        self.b = np.random.uniform(0, self.w, size=self.N)

        
    def GetIndex(self, vec):

        expand_vec = np.pad(vec, (0, self.piece_length-vec.shape[0]), 'constant')
        expand_vec = np.tile(expand_vec, self.piece_number)
        for i in range(self.rot_num):
            tmp = np.multiply(self.flip[i], expand_vec)
            ffht.fht(tmp)
        h = tmp.reshape(-1)[:self.N]
        h = np.floor((h + self.b) / self.w)
        return np.dot(self.lsh_r, h).astype(np.int)

class CSCFilter:
    def __init__(self, num_sets:int, hash_num:int, dim:int, total_capacity:int, w:float):
        print('Start initializing...')
        self.hash_num = hash_num
        self.num_sets = num_sets
        self.dim = dim
        self.total_capacity = total_capacity
        self.w = w

        self.single_capacity = int(np.ceil(self.total_capacity / self.num_sets / self.hash_num))
        self.cpbf = [ bitarray.bitarray([False]) * (self.num_sets * self.single_capacity) for _ in range(self.hash_num)]
        self.hash_func = HashFunc(self.hash_num, self.dim, self.w)

        print('Initialization finished!')
        self.visited =  [0] * self.num_sets * self.single_capacity
        self.count = 0
        self.flag = 0
        self.insert_time = 0

    def Insert(self, SetID:int, vector):
        t = time.time()
        locations = np.squeeze(self.hash_func.GetIndex(vector))
        for i in range(self.hash_num):
            location = (locations[i] + SetID) % (self.single_capacity * self.num_sets)
            self.cpbf[i][location] = True
            if i == 0:
                if self.visited[location] == 0:
                    self.visited[location] = 1
                    self.count += 1
        self.insert_time += time.time()-t

    def Query(self, vector, t):
        if self.flag == 0:
            self.flag = 1
            print(self.count)
            print(self.insert_time)
            print("")

        or_array = [bitarray.bitarray([False]) * self.num_sets for _ in range(t)]
        and_array = bitarray.bitarray(self.num_sets); and_array.setall(True)

        t0 = time.time()
        locations = np.squeeze(self.hash_func.GetIndex(vector)) % (self.single_capacity * self.num_sets)
        mlen = self.single_capacity*self.num_sets
        for i in range(self.hash_num):
            location = locations[i]
            if location + self.num_sets <= mlen:
                or_array[i % t] |= self.cpbf[i][location:location+self.num_sets]
            else:
                or_array[i % t][0:mlen-location] |= self.cpbf[i][location:mlen]
                or_array[i % t][mlen-location:] |= self.cpbf[i][0:self.num_sets+location-mlen]
        for i in range(t):
            and_array &= or_array[i]
        results = np.where(and_array)[0].tolist()
        t2 = time.time()

        return results, t2-t0

