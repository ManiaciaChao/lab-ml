from dataset_eng import init
from Sample import Sample
from utils import split_dataset
import numpy as np
from math import log2
from time import time
from multiprocessing import Pool


class NaiveBayes:
    def __init__(self):
        self.all_words_set = set()
        self.all_words_dict = {}
        self.all_words = []
        self.documents = {"ham": [], "spam": []}
        self.training = []

    def fit(self, training):
        self.training = training
        for i, sample in enumerate(training):
            self.all_words_set = self.all_words_set | set(sample.words)
        self.all_words = list(self.all_words_set)
        self.all_words_dict = {k: v for v, k in enumerate(self.all_words_set)}
        for i, sample in enumerate(training):
            print((i+1)/len(training))
            self.documents[sample.label].append(self.bow_gen(sample.words))

    def bow_gen(self, words):
        length = len(self.all_words)
        bow = [0] * length
        for word in words:
            if word in self.all_words_set:
                bow[self.all_words_dict[word]] += 1
        return np.array(bow)

    def prob(self, bow, label):
        types = self.documents[label]
        pi1 = len(types) / len(self.training)
        pi1 = log2(pi1)
        lenbow = len(bow)
        res = np.zeros(lenbow)
        for i in range(lenbow):
            # yki = (filter(lambda x: x[i] == bow[i], types))
            # print(len(list(yki)))
            yki = [x for x in types if x[i] == bow[i]]
            res[i] = (len(yki)+1) / (len(types)+2)
            # aaa = log2(aaa)
            # pi1 += aaa
        return pi1 + np.sum(np.log(res))

    def judge(self, words):
        bow = self.bow_gen(words)
        pi1 = self.prob(bow, "ham")
        pi2 = self.prob(bow, "spam")
        return ("ham" if pi1 > pi2 else "spam")


samples, tests = split_dataset(init("enron_small"), 0.8)

nb = NaiveBayes()
print("fitting")
nb.fit(samples)
print("fitted")

workers = 6  # IMPORTANT: should be number of physical cores of your PC
test_size = len(tests)  # 10000
chunk_size = 10  # size of each chunk
chunks_num = 10  # total number of chunks
chunks = [
    tests[i:i + chunk_size] for i in range(0, test_size, chunk_size)
]  # split testing set into chunks for multi-process calculating


def bow_gen(words, input):
    bow = {}
    for word in input:
        if word in words:
            if word in bow:
                bow[word] += 1
            else:
                bow[word] = 1
    return bow


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
    for cid, d in enumerate(chunks[0:chunks_num]):
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
