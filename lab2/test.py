import numpy as np
from multiprocessing import Pool
from dataset_eng import init
from Sample import Sample
from utils import split_dataset, split_into_chunks
from NaiveBayes import NaiveBayes

# samples = init("enron_3000")
# tests = init("enron_all")
training_set, testing_set = split_dataset(init("enron_all"), 0.15)
print("size of training set %d" % len(training_set))
print("size of testing set %d" % len(testing_set))

nb = NaiveBayes()
nb.fit(training_set)
nb.train()

workers = 6  # IMPORTANT: should be number of physical cores of your PC
chunk_size = 100  # size of each chunk
chunks = list(split_into_chunks(testing_set, chunk_size))
chunks_num = len(chunks)
print("total", chunks_num, "chunks")


def process(chunk_id):
    prompt = 'chunk {}/{}'.format(chunk_id, chunks_num)
    print(prompt, "starts")
    tests = chunks[chunk_id]
    length = len(tests)
    misclassified = 0
    for test in tests:
        predict = nb.predict(test.words)
        if predict != test.label:
            misclassified += 1
    print(prompt, "ends with", misclassified / length)
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


print("misclassification rate %f" % run())
