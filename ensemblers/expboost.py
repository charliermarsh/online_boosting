"""
    An online boosting algorithm which mixes SmoothBoost with
    "Learning from Expert Advice" from Chen '12.
"""


class EXPBooster(object):

    def __init__(self, Learner, classes, M=10):
        self.M = M
        self.learners = [Learner(classes) for _ in range(self.M)]
        self.w = [1.0 for _ in range(self.M)]

    def composite_prediction(self, i, x):
        return sum(l.predict(x) for l in self.learners[:i]) / (i + 1)

    def update(self, features, label):
        for i in range(self.M):
            pred = self.composite_prediction(i, features)
            if pred * label <= 0:
                self.w[i] /= 2
        for i in range(self.M):
            self.learners[i].partial_fit(features, label)

    def predict(self, features):
        total = sum(self.w[i] * self.composite_prediction(i, features)
                    for i in range(self.M))
        if total >= 0:
            return 1.0
        else:
            return -1.0
