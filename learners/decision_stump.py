"""
    An incremental decision stump based on Utgoff [1997].
"""
from math import log
from sys import maxint
import numpy as np


def best_metric(X, y):
    (num_examples, n) = X.shape
    best_thresh = None
    best_attribute = None
    minimum = maxint

    for i in range(n):

        def best_metric_for_attribute(i):
            best_thresh = None
            minimum = maxint

            order = (-X[:, i]).argsort()
            Xi = X[order]
            yi = y[order]

            num_left = num_examples
            pos_left = sum(filter(lambda x: x == 1, yi))
            num_right = 0
            pos_right = 0

            j = 0
            while j < num_examples:
                if yi[j] == 1:
                    pos_left -= 1
                    pos_right += 1
                num_left -= 1
                num_right += 1
                while j < num_examples - 1 and Xi[(j, i)] == Xi[(j + 1, i)]:
                    if yi[j + 1] == 1:
                        pos_left -= 1
                        pos_right += 1
                    num_left -= 1
                    num_right += 1
                    j += 1

                def entropy(p, n, N):
                    if n == 0:
                        return 0
                    x = float(p) / n
                    y = float(n - p) / n
                    if x > 0:
                        x = x * log(x)
                    if y > 0:
                        y = y * log(y)
                    return -float(n) / N * (x + y)

                score = entropy(pos_left, num_left, num_examples) + \
                    entropy(pos_right, num_right, num_examples)

                if score < minimum:
                    best_thresh = Xi[(j, i)]
                    minimum = score
                j += 1

            return score, best_thresh

        score, thresh = best_metric_for_attribute(i)
        if score < minimum:
            minimum = score
            best_thresh = thresh
            best_attribute = i

    return (best_attribute, best_thresh)


class DecisionStump(object):

    def __init__(self, classes):
        self.classes = classes
        self.X = None
        self.y = None
        self.comparator = None

    def predict(self, x):
        if not self.comparator:
            return self.classes[0]

        if self.comparator(x):
            return 1.0
        else:
            return -1.0

    def partial_fit(self, x, y, sample_weight=1.0):
        if self.X is None and self.y is None:
            self.X = x.toarray()
            self.y = np.array([y])
        else:
            self.X = np.vstack((self.X, x.toarray()))
            self.y = np.hstack((self.y, y))

        (attribute, threshold) = best_metric(self.X, self.y)
        print attribute, threshold
        self.comparator = lambda x: x[(0, attribute)] >= threshold
