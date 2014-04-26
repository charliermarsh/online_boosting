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
        self.epsilon = [0 for i in range(self.M)]

    def update(self, features, label):
        self.N += 1.0
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

            self.epsilon[i] = float(self.wrongWeight[i]) / \
                (self.wrongWeight[i] + self.correctWeight[i])
            if self.epsilon[i] >= 0.5:
                for j in range(i + 1, self.M):
                    self.epsilon[j] = None
                return

    def predict(self, features):
        # If you haven't been updated, just guess
        if not self.N:
            return 1

        label_weights = defaultdict(int)

        for i, learner in enumerate(self.learners):
            if self.epsilon[i] is None:
                break

            if self.epsilon[i] == 1:
                beta = 0.000001
            elif self.epsilon[i]:
                beta = (1 - self.epsilon[i]) / self.epsilon[i]
            else:
                beta = 100000.0
            label = learner.predict(features)
            label_weights[label] += log(beta)

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
