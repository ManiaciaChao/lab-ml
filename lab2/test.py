from dataset_eng import init
from Sample import Sample
from utils import split_dataset
import numpy as np
from math import log2, floor
from time import time
from multiprocessing import Pool
from NaiveBayes import NaiveBayes


# samples = init("english_200")
samples, test = split_dataset(init("trec06c_less"), 1)
print(len(samples))
samples, tests = split_dataset(samples[0:1000], 0.8)

# tests = init("english_email")[40:]

nb = NaiveBayes()
print("fitting")
nb.fit(samples)
print("")

workers = 6  # IMPORTANT: should be number of physical cores of your PC
test_size = len(tests)  # 10000
chunk_size = 8  # size of each chunk
chunks_num = 24  # total number of chunks
chunks = [
    tests[i:i + chunk_size] for i in range(0, test_size, chunk_size)
]  # split testing set into chunks for multi-process calculating


def process(chunk_id):
    print("chunk", chunk_id, "starts")
    tests = chunks[chunk_id]
    length = len(tests)
    misclassified = 0
    for test in tests:
        predict = nb.judge(test.words)
        # print(predict, test.label)
        if predict != test.label:
            misclassified += 1
    print("chunk", chunk_id, "ends with", misclassified / length)
    return misclassified


def run():
    # multi-process
    pool = Pool(processes=workers)
    # pool = Pool()  # use all core by default
    result = []
    # mapping
    for cid in range(chunks_num):
        # the reason why pass chunk_id instead of chunk to process() here
        # is to avoid additional memory copy caused
        result.append(pool.apply_async(process, args=(cid,)))
    pool.close()
    pool.join()
    # reducing
    total_misclassified = 0  # number of total misclassified test case
    for i in result:
        total_misclassified += i.get()
    percent = total_misclassified / (chunks_num * chunk_size)
    return percent


print(run())

# misclassified = 0
# for test in tests:
#     predict = nb.judge(test.words)
#     print(predict, test.label)
#     if predict != test.label:
#         misclassified += 1
# print(misclassified / len(tests))
