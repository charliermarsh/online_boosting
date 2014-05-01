from collections import defaultdict
import numpy as np


class OSBooster(object):

    def __init__(self, Learner, classes, M=10, g=0.10):
        self.M = M
        self.learners = [Learner(classes) for _ in range(self.M)]
        self.alpha = np.ones(self.M) / self.M
        self.gamma = g
        self.theta = self.gamma / (2 + self.gamma)

    def update(self, features, label):
        zt = 0.0
        w = 1.0
        for learner in self.learners:
            zt += learner.predict(features) * label - self.theta
            learner.partial_fit(features, label, sample_weight=w)
            if zt <= 0:
                w = 1.0
            else:
                w = (1.0 - self.gamma) ** (zt / 2.0)

    def raw_predict(self, features):
        return sum(learner.predict(features) for learner in self.learners)

    def predict(self, features):
        label_weights = defaultdict(int)
        for i in range(self.M):
            label = self.learners[i].predict(features)
            label_weights[label] += self.alpha[i]

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
