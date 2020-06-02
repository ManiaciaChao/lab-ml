import numpy as np
from lab1.KNN import KNNClassifier
import mnist
from multiprocessing import Pool
from matplotlib import pyplot as plot
from time import time

# mnist.init()
train_images, train_labels, test_images, test_labels = map(lambda x: x.astype(np.int16), mnist.load())
workers = 6
test_size = test_images.shape[0]
chunk_size = 20
chunks_num = 12
chunks = [
    [test_images[i:i + chunk_size], test_labels[i:i + chunk_size]
     ] for i in range(0, test_size, chunk_size)
]


def process(knn, chunk, chunk_id):
    print("chunk", chunk_id, "starts")
    images, labels = chunk;
    length = images.shape[0]
    misclassified = 0;
    for i in range(length):
        x = np.array([images[i]])
        res = knn.predict(x)[0]
        # print(i,res, labels[i])
        if res != labels[i]:
            misclassified += 1
    print("chunk", chunk_id, "ends with", misclassified / length)
    return misclassified


def run_knn(k):
    knn = KNNClassifier(k)
    knn.fit(train_images, train_labels)

    pool = Pool(processes=workers)
    result = []
    for cid, chunk in enumerate(chunks[0:chunks_num]):
        result.append(pool.apply_async(process, args=(knn, chunk, cid)))
    pool.close()
    pool.join()

    total_misclassified = 0
    for i in result:
        total_misclassified += i.get()
    percent = total_misclassified / (chunks_num * chunk_size);
    return percent


def make_plot(x_list, y_list):
    plot.figure('misclassification rate')
    plot.title("misclassification rate with different k")
    ax = plot.gca()
    ax.set_xlabel('k')
    ax.set_ylabel('misclassification rate')
    # ax.set_ylim(bottom=0,top=20)
    ax.plot(x_list, y_list, color='b', linewidth=1, alpha=0.6)
    ax.scatter(x_list, y_list, c='b', s=20, alpha=0.5)
    plot.show()


how_many_k = 20;
percents = []
percents_range = range(2, 1 + how_many_k)
start_time = time()
for i in percents_range:
    percent = run_knn(i)
    end_time = time()
    print('k={:d}, mis_rate={:f}, t={:f}'.format(i, percent,end_time-start_time))
    percents.append(percent)

end_time = time()
print(end_time-start_time)

make_plot(percents_range, percents)
