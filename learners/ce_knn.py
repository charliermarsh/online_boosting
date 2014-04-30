"""
    An online (error-correct) kNN algorithm based on Foerster.
"""

from collections import defaultdict
import heapq


def _inc(w):
    return 2 - (w - 2) * (w - 2) / 2


def _dec(w):
    return w * w / 2


class kNN(object):

    def __init__(self):
        self.num_points = 0
        self.LR = 0.1
        self.r = 0.05
        self.threshold = 0.1
        self.delta = 13
        self.weights = {}
        self.label_counts = defaultdict(int)

    def get_k_nearest(self, x, k):
        k = max(k, 1)

        def dist(y):
            return sum((x - y) * (x - y) for (x, y) in zip(x, y))

        pq = []
        for (y, c) in self.weights:
            w = self.weights[(y, c)]
            tagged_y = (-dist(y), y, c, w)
            if len(pq) < k:
                heapq.heappush(pq, tagged_y)
            else:
                heapq.heappushpop(pq, tagged_y)

        return {(y, c): w for (_, y, c, w) in pq}

    def update(self, example, label):
        k_nearest = self.get_k_nearest(
            example, self.LR * self.num_points)

        if self.predict(example, k_nearest=k_nearest) == label:
            for (y, c) in k_nearest:
                if c == label:
                    w = self.weights[(y, c)]
                    self.weights[(y, c)] = _inc(w)
        elif self.label_counts[label] < self.delta:

            for (y, c) in k_nearest:
                if c == label:
                    w = self.weights[(y, c)]
                    self.weights[(y, c)] = _dec(w)

            self.weights = dict([((y, c), w)
                                for (y, c), w in self.weights.iteritems() if w >= self.threshold])

        self.label_counts[label] += 1
        self.weights[(tuple(example), label)] = 1
        self.num_points += 1

    def predict(self, x, k_nearest=None):
        if not self.num_points:
            return 1

        if not k_nearest:
            k_nearest = self.get_k_nearest(x, self.r * self.num_points)

        label_weights = defaultdict(int)
        for (y, c) in k_nearest:
            label_weights[c] += self.weights[(y, c)]

        return max(label_weights.iterkeys(), key=(lambda key: label_weights[key]))
