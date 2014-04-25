from math import log, e
from sys import maxint
from random import random


class OGBooster(object):

    @staticmethod
    def loss(x):
        return log(1.0 + e ** (-x))

    @staticmethod
    def dloss(x):
        return - 1.0 / (1.0 + e ** x)

    def __init__(self, Learner, M=10, K=10):
        self.M = M
        self.K = K
        self.learners = [[Learner() for _ in range(self.K)]
                         for _ in range(self.M)]
        self.errors = [[0.0 for _ in range(self.K)]
                       for _ in range(self.M)]
        self.w = []
        self.f = [learners[0] for learners in self.learners]

    def update(self, features, label):
        w = -OGBooster.dloss(0)
        F = [0 for _ in range(self.M)]

        for m in range(self.M):
            # Track best of the K learners for this selector
            best_k = None
            min_error = maxint
            for k in range(self.K):
                h = self.learners[m][k]
                for _ in range(1 + int(10 * w)):
                    h.update(features, label)

                # Update error weight
                if h.predict(features) != label:
                    self.errors[m][k] += w

                if self.errors[m][k] < min_error:
                    min_error = self.errors[m][k]
                    best_k = k

            # Representative for selector M is best learner
            self.f[m] = self.learners[m][best_k]
            F[m] = self.f[m].predict(features)
            if m > 0:
                F[m] += F[m - 1]

            # Update weight using loss function
            w = - OGBooster.dloss(label * F[m])

    def predict(self, features):
        F = sum(h.predict(features) for h in self.f)
        p1 = (e ** F) / (1 + e ** F)
        if random() < p1:
            return 1
        return -1
