"""
    An online boosting algorithm which mixes SmoothBoost with
    "Learning from Expert Advice" from Chen '12.
"""

from random import random
from collections import defaultdict
import numpy as np
from osboost import OSBooster


def choose(p):
    r = random()
    n = len(p)
    p /= sum(p)
    cdf = 0.0
    for i in range(n):
        cdf += p[i]
        if r < cdf:
            return i + 1
    return n


class EXPBooster(object):

    def __init__(self, Learner, classes, M=10):
        self.M = M
        self.learners = [Learner(classes) for _ in range(self.M)]
        self.alpha = np.ones(self.M) / self.M

    def update(self, features, label):
        beta = 0.5
        exp_predict = 0.0

        for i, learner in enumerate(self.learners):
            exp_predict += learner.predict(features)
            if exp_predict * label <= 0:
                self.alpha[i] *= beta

        OSBooster.update_learners(features, label, self.learners)

    def predict(self, features):
        k = choose(self.alpha)
        label_weights = defaultdict(int)
        for i in range(k):
            label = self.learners[i].predict(features)
            label_weights[label] += self.alpha[i]

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
