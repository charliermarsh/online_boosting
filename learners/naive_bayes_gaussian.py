from math import log, e, pi
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
        self.mean = 1e-16 * np.ones((2, self.dim))
        self.M2 = 1e-16 * np.ones((2, self.dim))
        self.pc = 1e-16 * np.ones(2)

    def partial_fit(self, x, y, sample_weight=1.0):
        if self.dim is None:
            self.reset(x)

        if y < 1.0:
            y = 0
        else:
            y = 1

        prev_sum_w = self.sum_w
        self.sum_w += sample_weight
        self.pc[y] += sample_weight

        for i in range(self.dim):
            v = x[(0, i)]
            delta = v - self.mean[y][i]
            R = delta * sample_weight / self.sum_w
            self.mean[y][i] += R
            if prev_sum_w > 0.0:
                self.M2[y][i] += prev_sum_w * delta * R

    def raw_predict(self, x):
        if self.dim is None:
            self.reset(x)

        if self.sum_w < 1e-15:
            return 0.0

        prob = []
        for c in (0, 1):
            p = self.pc[c] / self.sum_w
            if p < 1e-15:
                if self.pc[1 - c] / self.sum_w < 1e-15:
                    return 0.0
                else:
                    return 1.0 - 2 * c
            p = log(p)
            mean_c = self.mean[c]
            M2_c = self.M2[c]

            for i in range(self.dim):
                if self.sum_w > 1.0:
                    var = M2_c[i] / (self.sum_w - 1.0)
                else:
                    var = 1e-16
                tmp = x[(0, i)] - mean_c[i]
                pfc = -0.5 * (log(2 * pi * var) + tmp * tmp / var)
                p += pfc

            prob.append(p)
        prob[1] = 1.0 / (1.0 + e ** (prob[0] - prob[1]))
        return 2.0 * prob[1] - 1.0

    def predict(self, x):
        if self.raw_predict(x) > 0.0:
            return 1.0
        return -1.0
