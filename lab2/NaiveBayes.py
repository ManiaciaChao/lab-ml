import numpy as np
from math import log2, floor


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
            print("\rprogress: %.1f%%" % ((i+1)/len(training)*100), end='')
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
