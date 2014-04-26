from math import log
from collections import defaultdict


class NaiveBayes(object):

    def __init__(self):
        self.total_count = 0
        self.label_counts = defaultdict(int)
        # f[feature][value of feature] -> int
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        # f[(feature, label)][value of feature] -> int
        self.feature_label_counts = defaultdict(lambda: defaultdict(int))
        self.labels = []

    def feature_given_label(self, feature, value, label, LOG=True):
        num = float(1 + self.feature_label_counts[(feature, label)][value])
        denom = float(2 + self.feature_counts[feature][value])
        if LOG:
            return log(num / denom)
        return num / denom

    def label_prior(self, label, LOG=True):
        num = float(1 + self.label_counts[label])
        denom = float(len(self.labels) + self.total_count)
        if LOG:
            return log(num / denom)
        return num / denom

    def prob_given_label(self, x, label, LOG=True):
        return self.label_prior(label) + sum(self.feature_given_label(feature, value, label) for feature, value in enumerate(x))

    def predict(self, x):
        if not self.labels:
            return 0

        probs = [self.prob_given_label(x, label) for label in self.labels]
        return self.labels[probs.index(max(probs))]

    def update(self, example, label):
        if not label in self.labels:
            self.labels.append(label)

        self.total_count += 1
        self.label_counts[label] += 1
        for feature, value in enumerate(example):
            self.feature_counts[feature][value] += 1
            self.feature_label_counts[(feature, label)][value] += 1
