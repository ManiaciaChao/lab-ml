from time import time
from os.path import join
from math import floor
from random import shuffle as random_shuffle
import numpy as np


def with_time(func, args=[]):
    start = time()
    res = func(*args)
    end = time()
    print(func.__name__, "takes", end-start)
    return res


def get_postfix(path: str):
    return "/".join(path.split("/")[-2:])


def split_dataset(dataset, ratio=0.8, shuffle=False):
    div = floor(len(dataset) * ratio)
    if (shuffle):
        random_shuffle(dataset)
    return dataset[0:div], dataset[div:]


def split_into_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def shuffle_split(data, labels, ratio=0.8, shuffle=True):
    # dataset = [data, labels]
    x, y = data, labels
    div = floor(len(data) * ratio)
    if shuffle:
        idx = np.random.permutation(len(data))
        x, y = data[idx], labels[idx]
    # training_data, testing_data, training_labels,testing_labels
    return x[0:div], x[div:], y[0:div], y[div:]
