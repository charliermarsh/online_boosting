from collections import defaultdict
from math import log
from numpy.random import poisson


class AdaBooster(object):

    def __init__(self, Learner, M=10):
        self.M = M
        self.N = 0
        self.learners = [Learner() for i in range(self.M)]
        self.wrongWeight = [0 for i in range(self.M)]
        self.correctWeight = [0 for i in range(self.M)]

    def update(self, features, label):
        self.N += 1
        lam = 1.0
        for i, learner in enumerate(self.learners):
            k = poisson(lam)
            for _ in range(k):
                learner.update(features, label)

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

        beta = []
        for i, learner in enumerate(self.learners):
            epsilon = float(self.wrongWeight[i]) / \
                (self.wrongWeight[i] + self.correctWeight[i])
            if epsilon == 1:
                beta.append(1000.0)
            else:
                beta.append(epsilon / (1 - epsilon))

        label_weights = defaultdict(int)
        for b, learner in zip(beta, self.learners):
            label = learner.predict(features)
            if b == 0:
                label_weights[label] += 100000
            else:
                label_weights[label] += log(1 / b)

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
