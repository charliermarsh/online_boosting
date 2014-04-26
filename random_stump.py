from collections import defaultdict
from random import randint


class RandomStump(object):

    def __init__(self):
        self.label_counts = defaultdict(int)
        self.label_sums = defaultdict(int)
        self.feature = None

    def update(self, example, label):
        if self.feature is None:
            self.feature = randint(0, len(example) - 1)

        self.label_sums[label] += example[self.feature]
        self.label_counts[label] += 1

    def predict(self, x):
        if self.feature is None:
            return 0

        def mean(y):
            return float(self.label_sums[y]) / self.label_counts[y]

        means = [mean(y) for y in self.label_sums.keys()]
        if len(means) == 1:
            return self.label_sums.keys()[0]

        m0 = min(means[1], means[0])
        m1 = max(means[1], means[0])
        mid = m0 + float(m1 - m0) / 2
        if x[self.feature] < mid:
            return self.label_sums.keys()[means.index(m0)]
        else:
            return self.label_sums.keys()[means.index(m1)]