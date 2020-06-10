from time import time
from os.path import join
from math import floor
from random import shuffle
from Sample import Sample
import numpy as np


def with_time(func, args=[]):
    start = time()
    res = func(*args)
    end = time()
    print(func.__name__, "takes", end-start)
    return res


def get_postfix(path: str):
    return "/".join(path.split("/")[-2:])


def split_dataset(dataset=[Sample], rate=0.8):
    div = floor(len(dataset) * rate)
    shuffle(dataset)
    return dataset[0:div], dataset[div:]


def split_into_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
