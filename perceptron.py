def scale(x):
    norm = sum(xi * xi for xi in x)
    return [xi / norm for xi in x]


class Perceptron(object):

    def __init__(self):
        self.w = []

    def output(self, x):
        x = scale(x)
        return sum(p * q for p, q in zip(self.w, x))

    def predict(self, x):
        if not self.w:
            self.w = [0.0 for _ in range(len(x))]

        if self.output(x) >= 0.0:
            return +1.0
        else:
            return -1.0

    def update(self, x, y):
        if not self.w:
            self.w = [0.0 for _ in range(len(x))]

        if self.predict(x) != y:
            self.w = [w + y * xi for w, xi in zip(self.w, x)]
