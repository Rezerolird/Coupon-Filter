import time
from DSBFFilter import DSFilter
import sys
import numpy as np
from utils import *
import json

train_path = str(sys.argv[1])
test_path = str(sys.argv[2])
begin = int(sys.argv[3])
dense = int(sys.argv[4])
num_sets = int(sys.argv[5])
hash_num = int(sys.argv[6])
dim = int(sys.argv[7])
max_ele = int(sys.argv[8])
w = float(sys.argv[9])
threshold = int(sys.argv[10])
R = float(sys.argv[11])




print("Reading data ...")
if begin == 586:
    n, insert_vecs, labels = read_exc_data(train_path, dim, begin)
    qn, test_vecs, _ = read_exc_data(test_path, dim, begin)
elif begin == 877:
    n, insert_vecs, labels = read_rev_data(train_path, dim, begin, num_sets)
    qn, test_vecs, _ = read_rev_data(test_path, dim, begin, num_sets) 
elif dense == 1:
    n, insert_vecs, labels = read_dense_data(train_path, dim, begin)
    qn, test_vecs, _ = read_dense_data(test_path, dim, begin)
else:
    n, insert_vecs, labels = read_sparse_data(train_path, dim, begin)
    qn, test_vecs, _ = read_sparse_data(test_path, dim, begin)
print("Reading complete")



frame = DSFilter(num_sets=num_sets, hash_num=hash_num, dim=dim, max_ele=max_ele, w=w, threshold=threshold)
for i in range(n):
    for label in labels[i]:
        frame.Insert(SetID=label, vector=insert_vecs[i])
print("Insertion Complete")


total_time = 0
count = 0
wubao = 0
loubao = 0
bb = 0
ww = 0

for i in range(qn):
	gt = Ground(test_vecs[i], insert_vecs, labels, R)
	results1, time = frame.Query(test_vecs[i])
	count += 1
	total_time += time
	wubao += len(results1)-len(gt.intersection(set(results1)))
	loubao += len(gt)-len(gt.intersection(set(results1)))
	bb += len(gt)
	ww += len(results1)
	if count == 100:
		break

print(total_time/count)
print(1-wubao/ww)
print(1-loubao/bb)
print(wubao/count)
print(loubao/count)
print("\n")
