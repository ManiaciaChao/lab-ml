from collections import Counter
import numpy as np
from queue import PriorityQueue
from sklearn.neighbors import BallTree as SKBallTree

def first(q: PriorityQueue):
    return q.queue[0][1]


class Ball:
    def __init__(self, center, tuples, radius, left, right):
        self.center = center
        self.tuples = tuples
        self.radius = radius
        self.left = left
        self.right = right


# TODO: building a balltree
class BallTree:
    def __init__(self, data, labels):
        # tuple: map([...data, label])
        self.tuples = np.column_stack((data, labels))
        self.root = self.build(self.tuples)

    @staticmethod
    def build(tuples):
        if len(tuples) == 0:
            return None;
        if len(tuples) == 1:
            return Ball(center=tuples[0][:-1], tuples=tuples,
                        radius=0, left=None, right=None)
        aaa = np.all(tuples == tuples[0], axis=1)
        if np.all(aaa):
            return Ball(center=tuples[0][:-1], tuples=tuples,
                        radius=0, left=None, right=None)
        avg = np.mean(tuples[:, :-1], axis=0)
        n2_avg_list = [np.linalg.norm(pt - avg) for pt in tuples[:, :-1]]
        n2_avg_max = np.argmax(n2_avg_list)  # index
        root = Ball(center=avg, tuples=tuples,
                    radius=n2_avg_list[n2_avg_max], left=None, right=None)

        pt1 = tuples[n2_avg_max]
        n2_pt1_list = np.array([np.linalg.norm(pt - pt1[:-1]) for pt in tuples[:, :-1]])
        n2_pt1_list_sorted = np.argsort(n2_pt1_list)
        n2_pt1_max = n2_pt1_list_sorted[-1]  # index
        pt2 = tuples[n2_pt1_max]
        n2_pt2_list = np.array([np.linalg.norm(pt - pt2[:-1]) for pt in tuples[:, :-1]])

        mask = n2_pt1_list < n2_pt2_list
        pt1_pts = tuples[mask]
        pt2_pts = tuples[~mask]
        print(len(pt1_pts), len(pt2_pts))
        root.left = BallTree.build(pt1_pts)
        root.right = BallTree.build(pt2_pts)

        return root


class KNNClassifier:

    def __init__(self, k, type=None):
        assert k >= 1, "k should be more than 0"
        self._data = None
        self._labels = None
        self.k = k
        self.ball_tree = None
        self.type = type

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
        predict = self._predict_brute
        if self.type == "naive":
            predict = self._predict_ball_tree
        elif self.type == "sklearn":
            predict = self._predict_sklearn
        predict_res = [predict(x) for x in x_list]
        return np.array(predict_res)

    # brutal predict an element
    def _predict_brute(self, x):
        assert x.shape[0] == self._data.shape[1], \
            "the number of features should be equal"
        # if np.dtype is uint, then below could cause problems
        distances = [np.linalg.norm(x - train) for train in self._data]
        nearest = np.argsort(distances)  # argsort returns indexes of elements in ASC
        k_labels = [self._labels[i] for i in nearest[:self.k]]
        return Counter(k_labels).most_common(1)[0][0]  # return the most common label\

    # TODO: use ball tree to predict an element, which takes much less time
    def init_tree(self, type="sklearn"):
        if self.ball_tree is None:
            if type == "sklearn":
                self.ball_tree = SKBallTree(self._data)
            elif type == "naive":
                self.ball_tree = BallTree(self._data, self._labels)
            self.type = type

    def _predict_sklearn(self, t):
        assert self.ball_tree is not None, "init_tree() should be called first"
        dist, indices = self.ball_tree.query([t], self.k)
        # labels = map(lambda i: self._labels[i], indices)
        k_labels = [self._labels[i] for i in indices[0]]
        return Counter(k_labels).most_common(1)[0][0]  # return the most common label

    def _predict_ball_tree(self, t):
        assert self.ball_tree is not None, "init_tree() should be called first"
        q = PriorityQueue()
        q.put((1, np.array([np.inf for i in range(785)])))
        self._predict_ball_tree_helper(t, q, self.ball_tree.root)
        res = []

        while not q.empty():
            res.append(q.get()[-1][-1])

        return Counter(np.array(res)).most_common(1)[0][0]  # return the most common label

    def _predict_ball_tree_helper(self, t, q: PriorityQueue, b: Ball):
        distance_c_t = np.linalg.norm(t - b.center)
        if distance_c_t >= b.radius+np.linalg.norm(t - first(q)[:-1]):
            return
        elif b.left is None and b.right is None:
            for p in b.tuples:
                distance_p_t = np.linalg.norm(t - p[:-1])
                if distance_p_t < np.linalg.norm(t - first(q)[:-1]):
                    q.put((-distance_p_t, p))
                    if q.qsize() > self.k:
                        q.get()
        else:
            self._predict_ball_tree_helper(t, q, b.left)
            self._predict_ball_tree_helper(t, q, b.right)

        # print(q.qsize())

