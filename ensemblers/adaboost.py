"""
    An online AdaBoost implementation based on Oza & Russel.
"""

from collections import defaultdict
from math import log
from numpy.random import poisson


class AdaBooster(object):

    def __init__(self, Learner, classes, M=10):
        self.M = M
        self.N = 0
        self.learners = [Learner(classes) for i in range(self.M)]
        self.wrongWeight = [0 for i in range(self.M)]
        self.correctWeight = [0 for i in range(self.M)]
        self.epsilon = [0 for i in range(self.M)]

    def update(self, features, label):
        self.N += 1.0
        lam = 1.0
        for i, learner in enumerate(self.learners):
            k = poisson(lam)
            if not k:
                continue

            for _ in range(k):
                learner.partial_fit(features, label)

            if learner.predict(features) == label:
                self.correctWeight[i] += lam
                lam *= self.N / (2 * self.correctWeight[i])
            else:
                self.wrongWeight[i] += lam
                lam *= self.N / (2 * self.wrongWeight[i])

    def predict(self, features):
        # If you haven't been updated, just guess
        if not self.N:
            return 1

        label_weights = defaultdict(int)

        for i, learner in enumerate(self.learners):

            def get_classifier_weight(i):
                if not (self.wrongWeight[i] + self.correctWeight[i]):
                    return 0.0

                epsilon = float(self.wrongWeight[i]) / \
                    (self.wrongWeight[i] + self.correctWeight[i])
                if epsilon > 0.5:
                    return 0.0
                elif epsilon == 0.0:
                    epsilon = 0.00001

                beta = epsilon / (1.0 - epsilon)
                return log(1.0 / beta)

            weight = get_classifier_weight(i)
            if weight > 0.0:
                label = learner.predict(features)
                label_weights[label] += weight

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
