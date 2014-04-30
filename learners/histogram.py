"""
    Based on Saffari's "Online Random Forests".
    Assumes features are in [-1, 1].
"""

from math import log
from collections import defaultdict
from random import random
import numpy as np


class Histogram(object):

    def __init__(self):
        self.range = [-1, 1]
        self.bins = 100
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

    def partial_fit(self, x, y, sample_weight=1.0):
        self.bin_counts[self.to_bin(x)][y] += sample_weight
        self.label_counts[y] += sample_weight

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
        self.weights = np.array([random_weights(N) for _ in range(self.H)])
        self.histograms = [Histogram() for _ in range(self.H)]
        self.initialized = True

    def __init__(self, classes):
        self.initialized = False
        self.labels = classes
        self.label_counts = defaultdict(int)
        self.total_count = 0

    def partial_fit(self, example, label, sample_weight=1.0):
        if not self.initialized:
            self.initialize(example.shape[1])

        self.label_counts[label] += sample_weight
        self.total_count += sample_weight

        for (w, h) in zip(self.weights, self.histograms):
            val = example.dot(w)
            h.partial_fit(val, label, sample_weight=sample_weight)

    def prob_given_label(self, example, label):
        prob = 1.0
        for (w, h) in zip(self.weights, self.histograms):
            val = example.dot(w)
            prob *= h.prob(val, label)
        prior = float(self.label_counts[label] + 1) / \
            (len(self.label_counts) + self.total_count)
        return prior * prob

    def predict(self, x):
        if not self.initialized:
            self.initialize(x.shape[1])
            return self.labels[0]

        probs = [self.prob_given_label(x, label) for label in self.labels]
        return self.labels[probs.index(max(probs))]
