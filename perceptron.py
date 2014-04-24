from math import sqrt


def scale(x):
    norm = sqrt(sum(y * y for y in x))
    return [y / norm for y in x]


class Perceptron(object):
    def __init__(self):
        self.w = None

    def predict(self, x):
        if not self.w:
            self.w = [0 for i in range(len(x))]

        x = scale(x)
        if sum(p * q for p, q in zip(self.w, x)) > 0:
            return 1
        else:
            return -1

    def update(self, x, y):
        if self.predict(x) == y:
            return

        self.w = [p + y * q for p, q in zip(self.w, x)]
