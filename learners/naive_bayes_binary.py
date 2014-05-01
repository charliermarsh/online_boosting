import numpy as np


class NaiveBayes(object):

    def __init__(self, classes):
        if set(classes) != set([-1.0, 1.0]):
            raise ValueError

        self.sum_w = 0
        self.dim = None
        self.neg = None
        self.pos = None
        self.pc = None

    def reset(self, x):
        self.self_w = 2 * 1e-16
        self.dim = x.shape[1]
        self.neg = 1e-16 * np.ones((2, self.dim))
        self.pos = 1e-16 * np.ones((2, self.dim))
        self.pc = 1e-16 * np.ones(2)

    def partial_fit(self, x, y, sample_weight=1.0):
        if self.dim is None:
            self.reset(x)

        if y < 1.0:
            y = 0
        else:
            y = 1

        self.sum_w += sample_weight
        self.pc[y] += sample_weight

        for i in range(self.dim):
            if x[(0, i)] < 1e-15:
                self.neg[y][i] += sample_weight
            else:
                self.pos[y][i] += sample_weight

    def raw_predict(self, x):
        if self.dim is None:
            self.reset(x)

        if self.sum_w < 1e-15:
            return 0.0

        prob = []
        for c in (0, 1):
            p = 1e+300 * (self.pc[c] / self.sum_w)
            if p < 1e-15:
                if self.pc[1 - c] / self.sum_w < 1e-15:
                    return 0.0
                else:
                    return 1.0 - 2 * c
            for i in range(self.dim):
                if x[(0, i)] == 0.0:
                    p *= self.neg[c][i] / self.pc[c]
                else:
                    p *= self.pos[c][i] / self.pc[c]
            prob.append(p)
        s = sum(prob)
        if s > 1e-15:
            return 2.0 * (prob[1] / s) - 1.0
        return 0.0

    def predict(self, x):
        if self.raw_predict(x) > 0.0:
            return 1.0
        return -1.0
