from math import e
from collections import defaultdict


class SmoothBooster(object):

    def __init__(self, Learner, classes, M=10):
        self.M = M
        self.learners = [Learner(classes) for _ in range(self.M)]

    def update(self, features, label):
        F = 0
        w = 0.5

        for learner in self.learners:
            learner.partial_fit(features, label, sample_weight=w)
            F += learner.predict(features)
            w = 1.0 / (1.0 + e ** (label * F))

    def predict(self, features):
        label_weights = defaultdict(int)

        for i, learner in enumerate(self.learners):

            label = learner.predict(features)
            label_weights[label] += 1

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
