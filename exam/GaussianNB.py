import numpy as np
from math import log, pi
from multiprocessing import Pool
from scipy.stats import norm

# naive bayes mail classifier


class GaussianNB:
    def __init__(self, alpha=1):
        self.__dict = {}
        self.__labels = []
        self.__means = {}
        self.__vars = {}
        self.__training_size = 0
        self.__priors = {}
        self.__trained = False
        self.__alpha = alpha

    # gaussian
    def __estimate(self):
        for label in self.__labels:
            features_list = self.__dict[label]
            self.__means[label] = np.mean(features_list, axis=0)
            self.__vars[label] = np.var(features_list, axis=0)
            self.__priors[label] = features_list.shape[0] / \
                self.__training_size
        # return px, py

    def __prob(self, features, label: str):
        means = self.__means[label]
        variances_x2 = self.__vars[label] * 2
        sigma = np.sqrt(self.__vars[label])
        uniform = 1 / np.sqrt(variances_x2 * pi)
        prob = uniform*np.exp(-(np.power(features-means, 2) / variances_x2))
        prob = norm.pdf(features, means, sigma)
        # print(prob[prob > 1])
        return prob

    def fit(self, features_list, labels):
        # features_list = dataset[:, :-1]
        # labels = dataset[:, -1:].T[0]
        # initialize
        assert len(
            features_list) > 0, "size of training sets should be more than 0"
        assert len(features_list) == len(
            labels), "sizes of labels should be equal to features_list"
        self.__labels = list(set(labels))
        for label in labels:
            self.__dict[label] = features_list[labels == label]
        self.__training_size = len(features_list)

        self.__estimate()
        return self

    def predict(self, inputs):
        res = []
        for ele in inputs:
            res.append(self.__predict(ele))
        if len(res) == 1:
            return res[0]
        return res

    def __predict(self, input):
        assert self.__training_size > 0, "fit() should be called before train()"
        # calculate conditional probability
        probs = []
        for label in self.__labels:
            prob = np.sum(np.log(self.__prob(input, label))) + \
                np.log(self.__priors[label])
            probs.append(prob)
        return self.__labels[np.argmax(probs)]
