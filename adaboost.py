from collections import defaultdict
from math import log
from numpy.random import poisson
from naive_bayes import NaiveBayes as Learner


class AdaBooster(object):

    def __init__(self, N=20):
        self.N = N
        self.learners = [Learner() for i in range(self.N)]
        self.wrongWeight = [1 for i in range(self.N)]
        self.correctWeight = [1 for i in range(self.N)]

    def update(self, features, label):
        lam = 1.0
        for i, learner in enumerate(self.learners):
            k = poisson(lam=lam)
            for _ in range(k):
                learner.update(features, label)

            if learner.predict(features) == label:
                self.correctWeight[i] += lam
                lam *= self.N / (2 * self.correctWeight[i])
            else:
                self.wrongWeight[i] += lam
                lam *= self.N / (2 * self.wrongWeight[i])

    def predict(self, features):
        beta = []
        for i, learner in enumerate(self.learners):
            epsilon = float(self.wrongWeight[i]) / \
                (self.wrongWeight[i] + self.correctWeight[i])
            beta.append(epsilon / (1 - epsilon))

        label_weights = defaultdict(int)
        for b, leaner in zip(beta, self.learners):
            label = learner.predict(features)
            label_weights[label] += log(1 / b)

        return max(label_weights)
