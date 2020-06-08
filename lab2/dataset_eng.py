# init(): Sample[]

import string
import os
from os.path import exists, isfile, splitext, join
import pickle
import nltk
from nltk.corpus import stopwords
from Sample import Sample, WordList
from utils import with_time


def removePunctuation(raw: str):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    return raw.translate(table)


def removeStopWords(word_list: WordList):
    stopwords_list = [*stopwords.words('english'), "im"]
    filtered_words = [word for word in word_list if word not in stopwords_list]
    return filtered_words


def normalize(raw: str):
    lower = raw.lower()
    withoutPunc = removePunctuation(lower)
    withoutStopWords = removeStopWords(withoutPunc.split())
    return withoutStopWords


def readfiles(dirname: str = "english_email"):
    results = []
    subdirs = os.listdir(dirname)
    for subdir in subdirs:
        filenames = os.listdir(join(dirname, subdir))
        filenames_full = map(lambda f: join(dirname, subdir, f), filenames)
        txts = [x for x in filenames_full
                if isfile(x) and splitext(x)[1] == '.txt']
        for txt in txts:
            with open(txt, 'r') as f:
                try:
                    results.append(Sample(normalize(f.read()), subdir))
                except:
                    init()
                    print(txt)
    return results


def init():
    pkl_name = "eng.pkl"
    if not exists(pkl_name):
        nltk.download("stopwords")
        print("parsing files...")
        sets = readfiles()
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
    data = with_time(init)
    print(data[randint(0, len(data))].words)
