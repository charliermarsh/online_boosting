from math import e


def normalize(x):
    norm = float(sum(x))
    return [xi / norm for xi in x]


class Winnow(object):

    def __init__(self):
        self.w = []
        self.N = 0
        self.eta = 2.0

    def output(self, x):
        return sum(p * q for p, q in zip(self.w, x))

    def predict(self, x):
        if not self.w:
            self.N = len(x)
            self.w = [1.0 / self.N for _ in range(len(x))]

        if self.output(x) > 0:
            return 1.0
        else:
            return -1.0

    def update(self, x, y):
        if not self.w:
            self.N = len(x)
            self.w = [1.0 / self.N for _ in range(len(x))]

        if self.predict(x) != y:
            for i in range(len(self.w)):
                self.w[i] *= e ** (self.eta * y * x[i])
            self.w = normalize(self.w)
