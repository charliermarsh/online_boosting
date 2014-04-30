from math import e
import numpy as np


def normalize(x):
    norm = float(sum(x))
    return [xi / norm for xi in x]


class Winnow(object):

    def __init__(self, classes):
        self.initialized = False
        self.eta = 2.0

    def initialize(self, x):
        self.N = x.shape[1]
        self.w = np.array([1.0 / self.N for _ in range(self.N)])
        self.initialized = True

    def output(self, x):
        return x * self.w

    def predict(self, x):
        if not self.initialized:
            self.initialize(x)

        if self.output(x) > 0:
            return 1.0
        else:
            return -1.0

    def partial_fit(self, x, y, sample_weight=1.0):
        if not self.initialized:
            self.initialize(x)

        x_iter = x.toarray()[0]

        if self.predict(x) != y:
            coeff = np.array([e ** (self.eta * y * xi)
                             for xi in x_iter])
            self.w = np.multiply(self.w, coeff)
            self.w /= np.linalg.norm(self.w, ord=1)
