from time import time
from os.path import join
import numpy as np
from math import floor
from Sample import Sample


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
    return dataset[0:div], dataset[div:]
