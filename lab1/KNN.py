from collections import Counter
import numpy as np


# class Ball:
#     def __init__(self, center, data, label, radius):
#         self.center = center
#         self.data = data
#         self.label = label
#         self.radius = radius
#
#
# # TODO: building a balltree
# class BallTree:
#     def __init__(self, data, label):
#         self.data = data
#         self.label = label
#         self.root = self.build_tree(data, label)
#
#     def build_tree(self, data, label):
#         return 1


class KNNClassifier:

    def __init__(self, k):
        assert k >= 1, "k should be more than 0"
        self._data = None
        self._labels = None
        self.k = k

    def set_k(self, k):
        self.k = k

    # fit training data
    def fit(self, data, labels):
        assert data.shape[0] == labels.shape[0], \
            "the number of data and labels should be equal"
        assert self.k <= data.shape[0], "k should be less than total size of training data"
        self._data = data
        self._labels = labels
        return self

    def predict(self, x_list):
        assert self._data is not None and self._labels is not None, \
            "fit should be called first"
        assert x_list.shape[1] == self._data.shape[1], \
            "the number of features should be equal"
        predict_res = [self._predict_brute(x) for x in x_list]
        return np.array(predict_res)

    # TODO: use ball tree to predict an element, which takes much less time
    def _predict_ball_tree(self, x):
        return

    # brutal predict an element
    def _predict_brute(self, x):
        assert x.shape[0] == self._data.shape[1], \
            "the number of features should be equal"
        # if np.dtype is uint, then below could cause problems
        distances = [np.linalg.norm(x - train) for train in self._data]
        nearest = np.argsort(distances)  # argsort returns indexes of elements in ASC
        k_labels = [self._labels[i] for i in nearest[:self.k]]
        return Counter(k_labels).most_common(1)[0][0]  # return the most common label
