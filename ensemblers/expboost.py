"""
    An online boosting algorithm which mixes SmoothBoost with
    "Learning from Expert Advice" from Chen '12.
"""

from random import random
from collections import defaultdict
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


class EXPBooster(OSBooster):

    def update(self, features, label):
        beta = 0.5
        exp_predict = 0.0

        for i, learner in enumerate(self.learners):
            exp_predict += learner.predict(features)
            if exp_predict * label <= 0:
                self.alpha[i] *= beta

        super(EXPBooster, self).update(features, label)

    def predict(self, features):
        k = choose(self.alpha)
        label_weights = defaultdict(int)
        for i in range(k):
            label = self.learners[i].predict(features)
            label_weights[label] += self.alpha[i]

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
