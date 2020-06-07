import pickle
import numpy as np
import mnist
from multiprocessing import Pool
from matplotlib import pyplot as plt
from time import time
from os.path import exists
from KNN import KNNClassifier
from random import randrange

# IMPORTANT: mnist.init() should be called the first time you run this script
mnist.init()
# convert type of numpy array elements into int16, avoiding subtraction overflow
train_images, train_labels, test_images, test_labels = map(
    lambda x: x.astype(np.int16), mnist.load())

# initializing multi-process options
# in Unix-like OS, multiprocessing is implemented based on fork().
# to take advantage of COW, knn instance and chunks should be global.
# by using COW, memory cost can be reduced to 1/workers.
# so multiprocessing.SharedMemory is no longer needed xD
# however, such optimization may not work in Windows :(
knn = KNNClassifier(10)
knn.fit(train_images, train_labels)
workers = 6  # IMPORTANT: should be number of physical cores of your PC
test_size = test_images.shape[0]  # 10000
chunk_size = 10  # size of each chunk
chunks_num = 10  # total number of chunks
chunks = [
    [test_images[i:i + chunk_size], test_labels[i:i + chunk_size]
     ] for i in range(0, test_size, chunk_size)
]  # split testing set into chunks for multi-process calculating


# run KNN on a specific chunk
# knn is an instance of KNNClassifier
# pass k here to avoid multiprocess issues with Windows
def process(chunk_id, k):
    print("chunk", chunk_id, "starts")
    knn.set_k(k)
    images, labels = chunks[chunk_id]
    length = images.shape[0]
    misclassified = 0
    for i in range(length):
        x = np.array([images[i]])
        res = knn.predict(x)[0]
        # print(i,res, labels[i])
        if res != labels[i]:
            misclassified += 1
    print("chunk", chunk_id, "ends with", misclassified / length)
    return misclassified


# run KNN with specific k on all testing set
def run_knn(k):
    # multi-process
    pool = Pool(processes=workers)
    # pool = Pool()  # use all core by default
    result = []
    # mapping
    for cid in enumerate(chunks[0:chunks_num]):
        # the reason why pass chunk_id instead of chunk to process() here
        # is to avoid additional memory copy caused
        result.append(pool.apply_async(process, args=(cid, k)))
    pool.close()
    pool.join()
    # reducing
    total_misclassified = 0  # number of total misclassified test case
    for i in result:
        total_misclassified += i.get()
    percent = total_misclassified / (chunks_num * chunk_size)
    return percent


# draw plot of misclassification rate with different k
def make_plot(x_list, y_list):
    plt.figure('misclassification rate')
    plt.title("misclassification rate with different k")
    ax = plt.gca()
    ax.set_xlabel('k')
    ax.set_ylabel('misclassification rate')
    # ax.set_ylim(bottom=0,top=20)
    ax.plot(x_list, y_list, color='b', linewidth=1, alpha=0.6)
    ax.scatter(x_list, y_list, c='b', s=20, alpha=0.5)
    plt.show()


def init_tree(type="sklearn"):
    start_time = time()
    if exists(type+".pkl"):
        print(type + " dump found! loading...")
        knn.type = type
        with open(type+".pkl", 'rb') as f:
            knn.ball_tree = pickle.load(f)
    else:
        print(type + " dump not found! creating...")
        knn.init_tree(type)
        with open(type+".pkl", 'wb') as f:
            pickle.dump(knn.ball_tree, f)
    end_time = time()
    print("initializing takes {:f} sec".format(end_time - start_time))


def displayImages(data_list):
    size = 28
    fig = plt.figure()
    columns = len(data_list)
    rows = 1
    for index, data in enumerate(data_list):
        img = np.array(data, dtype='float').reshape((size, size))
        fig.add_subplot(rows, columns, index+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()


def show_most_similar(id):
    target = test_images[id]
    nearest_k = knn._nearest_k_brute(target)
    most_similar = [train_images[i] for i in nearest_k]
    displayImages([target, *most_similar])


def show_misrate():
    how_many_k = 20
    percents = []
    percents_range = range(1, 1 + how_many_k)
    start_time = time()
    for i in percents_range:
        percent = run_knn(i)
        end_time = time()
        print('k={:d}, mis_rate={:f}, t={:f}'.format(
            i, percent, end_time - start_time))
        percents.append(percent)

    end_time = time()
    print(end_time - start_time)
    make_plot(percents_range, percents)


# in Windows, multiprocess-related functions
# should always be called in __main__ scope
if __name__ == "__main__":
    init_tree()
    show_most_similar(randrange(test_size))
    # show_misrate()
