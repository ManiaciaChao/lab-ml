import numpy as np
from multiprocessing import Pool
from dataset_eng import init
from Sample import Sample
from utils import split_dataset, split_into_chunks
from NaiveBayes import NaiveBayes

# samples = init("enron_3000")
# tests = init("enron_all")
train_name = "english_email"
test_name = "enron_3000"
split_ratio = 0
# training_set, testing_set = split_dataset(init(train_name), split_ratio)
training_set = init(train_name)
testing_set = init(test_name)
print("size of training set %d" % len(training_set))
print("size of testing set %d" % len(testing_set))

nb = NaiveBayes()
nb.fit(training_set).train()

workers = 6  # IMPORTANT: should be number of physical cores of your PC
chunk_size = 100  # size of each chunk
chunks = list(split_into_chunks(testing_set, chunk_size))
chunks_num = len(chunks)
print("total", chunks_num, "chunks")


def process(chunk_id):
    prompt = 'chunk {}/{}'.format(chunk_id, chunks_num)
    print(prompt, "starts")
    tests = chunks[chunk_id]
    # here we assume that spam is positive
    TP = FP = TN = FN = 0
    for test in tests:
        predict = nb.predict(test.words)
        if test.label == predict:
            if predict == "spam":
                TP += 1
            else:
                TN += 1
        else:
            if predict == "spam":
                FP += 1
            else:
                FN += 1
    print(prompt, "ends with TP={} FP={} TN={} FN={}".format(
        TP, FP, TN, FN))
    return TP, FP, TN, FN


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
    total_TP = total_FP = total_TN = total_FN = 0
    for i in result:
        TP, FP, TN, FN = i.get()
        total_TP += TP
        total_FP += FP
        total_TN += TN
        total_FN += FN
    # percent = total_misclassified / (chunks_num * chunk_size)
    return total_TP, total_FP, total_TN, total_FN


TP, FP, TN, FN = run()
accuracy = (TP + TN) / (TP + FP + TN + FN)
recall = TP / (TP+FN)
precision = TP / (TP + FP)
f1_score = (2*recall*precision)/(recall+precision)

cli_output = """result
training name:  {train_name:s}
testing name:   {test_name:s}
split ratio:     {ratio:f}
training size:  {train_size:d}
testing size:   {test_size:d}
total words:    {total_words:d}
accuracy:       {accuracy:f}
recall:         {recall:f}
precision:      {precision:f}
f1_score:       {f1_score:f}
""".format(train_name=train_name, test_name=test_name, ratio=split_ratio,
           train_size=len(training_set), test_size=len(testing_set),
           total_words=nb.total_words,
           accuracy=accuracy, recall=recall,
           precision=precision, f1_score=f1_score)
csv_output = ("{train_name:s},""{test_name:s},""{ratio:f},""{train_size:d},""{test_size:d},"
              "{total_words:d},""{accuracy:f},""{recall:f},""{precision:f},""{f1_score:f}"
              ).format(train_name=train_name, test_name=test_name, ratio=split_ratio,
                       train_size=len(training_set), test_size=len(testing_set),
                       total_words=nb.total_words,
                       accuracy=accuracy, recall=recall,
                       precision=precision, f1_score=f1_score)
print(cli_output)
f = open("result.csv", "a")
f.write(csv_output+"\n")
f.close()
