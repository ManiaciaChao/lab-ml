import numpy as np
from math import log
from multiprocessing import Pool


# naive bayes mail classifier
class NaiveBayes:
    def __init__(self, alpha=1):
        self.__all_words = {}
        self.__documents = {"ham": [], "spam": []}
        self.__training_size = 0
        self.__px_spam = 0
        self.__px_ham = 0
        self.__py_spam = 0
        self.__py_ham = 0
        self.__trained = False
        self.__alpha = alpha

    def __bow_gen(self, words):
        assert len(words) > 0, "size of words list should be more than 0"
        length = len(self.__all_words)
        bow = np.zeros(length)
        for word in words:
            if word in self.__all_words:
                bow[self.__all_words[word]] += 1
        return bow

    def __estimate(self, label):
        assert label == "ham" or "spam", "label should be \"ham\" or \"spam\""
        # select all emails with given label
        docs = np.array(self.__documents[label])
        px = np.sum(docs, axis=0)  # BOW of all emails with given label
        py = docs.shape[0]  # count of emails with given label
        words = np.sum(docs)  # count of all words with given label
        # estimate conditional probability with laplace smoothing
        px = np.log((px+self.__alpha) /
                    (words+self.__alpha*len(self.__all_words)))
        # estimate prior probability
        py = log(py/self.__training_size)
        return px, py

    # generate SOW for all emails and BOW for each email
    def fit(self, training):
        # initialize
        assert len(training) > 0, "size of training sets should be more than 0"
        self.__training_size = len(training)
        print("training size:", self.__training_size)

        # generate SOW
        all_words_set = set()
        for i, sample in enumerate(training):
            print("\rgenerating SOW: %.1f%%" %
                  ((i+1)/self.__training_size*100), end='')
            all_words_set = all_words_set | set(sample.words)  # conjunction
        print(" total words:", len(self.__all_words))

        # turn SOW into a map in the from of value->key,
        # improving performance of constructing BOW
        self.__all_words = {k: v for v, k in enumerate(all_words_set)}

        # classify all emails
        for i, sample in enumerate(training):
            print("\rgenerating BOW: %.1f%%" %
                  ((i+1)/self.__training_size*100), end='')
            self.__documents[sample.label].append(self.__bow_gen(sample.words))
        print("")  # print an EOL

    def train(self):
        assert self.__training_size > 0, "fit() should be called before train()"
        # polynomial parameter estimation
        self.__px_spam, self.__py_spam = self.__estimate("spam")
        self.__px_ham, self.__py_ham = self.__estimate("ham")
        self.__trained = True

    def predict(self, input):
        assert self.__trained is True, "train() should be called before predict()"
        # calculate conditional probability
        bow = self.__bow_gen(input)
        spam = np.dot(bow, self.__px_spam) + self.__py_spam
        ham = np.dot(bow, self.__px_ham) + self.__py_ham
        return ("ham" if spam < ham else "spam")
