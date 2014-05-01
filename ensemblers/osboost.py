from collections import defaultdict


class OSBooster(object):

    def __init__(self, Learner, classes, M=10, g=0.10):
        self.M = M
        self.learners = [Learner(classes) for _ in range(self.M)]
        self.alpha = [1.0 / M for _ in range(self.M)]
        self.g = g

    def update(self, features, label):
        OSBooster.update_learners(features, label, self.learners, g=self.g)

    @staticmethod
    def update_learners(features, label, learners, g=0.10):
        gamma = g
        theta = g / (2 + g)
        zt = 0.0
        w = 1.0
        for learner in learners:
            zt += learner.predict(features) * label - theta
            learner.partial_fit(features, label, sample_weight=w)
            if zt <= 0:
                w = 1.0
            else:
                w = (1.0 - gamma) ** (zt / 2.0)

    def predict(self, features):
        label_weights = defaultdict(int)
        for i in range(self.M):
            label = self.learners[i].predict(features)
            label_weights[label] += self.alpha[i]

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
