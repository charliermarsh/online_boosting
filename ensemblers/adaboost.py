"""
    An online AdaBoost implementation based on Oza & Russel.
"""

from collections import defaultdict
from math import log
from numpy.random import poisson, seed

seed(0)


class AdaBooster(object):

    def __init__(self, Learner, classes, M=10):
        self.M = M
        self.learners = [Learner(classes) for i in range(self.M)]
        self.wrongWeight = [0 for i in range(self.M)]
        self.correctWeight = [0 for i in range(self.M)]
        self.epsilon = [0 for i in range(self.M)]

    def update(self, features, label):
        lam = 1.0
        for i, learner in enumerate(self.learners):
            k = poisson(lam)
            if not k:
                continue

            for _ in range(k):
                learner.partial_fit(features, label)

            if learner.predict(features) == label:
                self.correctWeight[i] += lam
                lam *= (self.correctWeight[i] + self.wrongWeight[i]) / \
                    (2 * self.correctWeight[i])
            else:
                self.wrongWeight[i] += lam
                lam *= (self.correctWeight[i] + self.wrongWeight[i]) / \
                    (2 * self.wrongWeight[i])

    def predict(self, features):
        label_weights = defaultdict(int)

        for i, learner in enumerate(self.learners):

            epsilon = (self.correctWeight[i] + 1e-16) / \
                (self.wrongWeight[i] + 1e-16)
            weight = log(epsilon)
            label = learner.predict(features)
            label_weights[label] += weight

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
