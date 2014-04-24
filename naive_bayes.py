from math import log
from collections import defaultdict


class NaiveBayes(object):

    def __init__(self):
        self.counts = defaultdict(int)
        self.priors = defaultdict(int)
        self.seen = 0
        self.labelPriors = defaultdict(int)
        self.labels = [-1.0, +1.0]

    def probability(self, feature, value, label):
        return float(1 + self.counts[(feature, value, label)]) / (2 + self.priors[feature, value])

    def labelPrior(self, label):
        if not self.seen:
            return 1
        return float(self.labelPriors[label] + 1) / self.seen

    def predict(self, x):
        max_val = None
        max_label = None
        for label in self.labels:
            prob = sum(log(self.probability(i, value, label))
                       for i, value in enumerate(x))
            prob += log(self.labelPrior(label))
            if not max_val or prob > max_val:
                max_val = prob
                max_label = label
        return max_label

    def update(self, x, y):
        self.seen += 1
        self.labelPriors[y] += 1
        for i, j in enumerate(x):
            self.priors[(i, j)] += 1
            self.counts[(i, j, y)] += 1
