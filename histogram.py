"""
    Assumes features are in [-1, 1]
"""

from math import log
from collections import defaultdict
from random import random


class Histogram(object):

    def __init__(self):
        self.range = [-1, 1]
        self.bins = 4
        self.bin_size = float(self.range[1] - self.range[0]) / self.bins
        self.bin_counts = [defaultdict(int) for i in range(self.bins)]
        self.label_counts = defaultdict(int)

    def to_bin(self, x):
        if x > self.range[1] or x < self.range[0]:
            raise ValueError(x)

        # Bins are [x, y), but right end-point is valid
        if x == self.range[1]:
            return self.bins - 1

        dist = x - self.range[0]
        return int(dist / self.bin_size)

    def update(self, x, y):
        self.bin_counts[self.to_bin(x)][y] += 1
        self.label_counts[y] += 1

    def prob(self, x, y):
        if not self.label_counts:
            return 1.0

        num = float(1 + self.bin_counts[self.to_bin(x)][y])
        denom = float(len(self.label_counts) + self.label_counts[y])
        return num / denom


def random_weights(N):
    def flip(i):
        if random() > 0.50:
            return -i
        return i
    v = [random() for _ in range(N)]
    norm = sum(v)
    return [flip(i) / norm for i in v]


class RNB(object):

    def initialize(self, N):
        self.H = int(log(N))
        self.weights = [random_weights(N) for _ in range(self.H)]
        self.histograms = [Histogram() for _ in range(self.H)]
        self.initialized = True

    def __init__(self):
        self.initialized = False
        self.labels = [-1.0, +1.0]
        self.label_counts = defaultdict(int)
        self.total_count = 0

    def update(self, example, label):
        if not self.initialized:
            self.initialize(len(example))

        self.label_counts[label] += 1
        self.total_count += 1

        for (w, h) in zip(self.weights, self.histograms):
            val = sum(p * q for p, q in zip(w, example))
            h.update(val, label)

    def prob_given_label(self, example, label):
        prob = 1.0
        for (w, h) in zip(self.weights, self.histograms):
            val = sum(p * q for p, q in zip(w, example))
            prob *= h.prob(val, label)
        prior = float(self.label_counts[label] + 1) / \
            (len(self.label_counts) + self.total_count)
        return prior * prob

    def predict(self, x):
        if not self.initialized:
            self.initialize(len(x))
            return self.labels[0]

        probs = [self.prob_given_label(x, label) for label in self.labels]
        return self.labels[probs.index(max(probs))]
