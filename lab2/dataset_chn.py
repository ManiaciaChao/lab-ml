# init(): Sample[]

import string
import os
from os.path import exists, isfile, splitext, join
import pickle
from Sample import Sample, Labels, WordList
import jieba
import re
import email
from multiprocessing import Pool
from utils import with_time, get_postfix


def removeStopWords(words: WordList, stopwords: WordList):
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words


def normalize(raw_str: string, stopwords: WordList):
    text = "".join(raw_str.split())
    non_chn = re.compile(r"\W+|\d+|[a-z]+")
    text = "".join(non_chn.split(text))
    seg_list = jieba.lcut(text)
    return removeStopWords(seg_list, stopwords)


def parseMailBundle(subdirname: string, stopwords: WordList, labels: Labels):
    print(subdirname, "start!")
    files = os.listdir(subdirname)
    results = []
    for data_file in files:
        filename = join(subdirname, data_file)
        with open(filename, 'r', encoding="gbk") as f:
            try:
                words = normalize(email.message_from_file(
                    f).get_payload(), stopwords)
                label = labels[get_postfix(filename)]
                results.append(Sample(words, label))
            except:
                pass
    print(subdirname, "end!")
    return results


def read_labels(dirname: string = "trec06c"):
    filename = join(dirname, "full", "index")
    result = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            labels, path = line.strip().split()
            key = get_postfix(path)
            result[key] = labels
    return result


def readfiles(dirname: string = "trec06c"):
    labels = read_labels(dirname)
    results = []

    stopwords = read_all_stopwords()
    dirname = join(dirname, "data")
    subdirs = os.listdir(dirname)
    subdirs.sort()

    jieba.initialize()
    workers = 6
    pool = Pool(processes=workers)
    pool_results = []

    for subdir in subdirs:
        subdirname = join(dirname, subdir)
        pool_results.append(pool.apply_async(
            parseMailBundle, args=(subdirname, stopwords, labels)))
    pool.close()
    pool.join()

    for bundle_result in pool_results:
        b = bundle_result.get()
        for result in b:
            results.append(result)

    return results


def read_all_stopwords(dirname: string = "stopwords"):
    stopwords = []
    for file in os.listdir(dirname):
        stopwords += read_stopwords(join(dirname, file))
    return stopwords


def read_stopwords(filename: string):
    stopwords = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def init(filename: string = "trec06c") -> [Sample]:
    pkl_name = filename + ".pkl"
    if not exists(pkl_name):
        print("parsing files...")
        sets = readfiles(filename)
        with open(pkl_name, 'wb') as f:
            print("saving dump...")
            pickle.dump(sets, f)
        print("done!")
        return sets
    else:
        with open(pkl_name, 'rb') as f:
            print("reading dump...")
            sets = pickle.load(f)
        print("done!")
        return sets


if __name__ == "__main__":
    from random import randint
    data = with_time(init, ["trec06c"])
    print(data[randint(0, len(data))].words)
