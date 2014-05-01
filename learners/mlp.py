import numpy as np
import neurolab as nl


class MLP(object):

    def __init__(self, classes):
        self.model = None
        self.classes = classes

    def reset(self, x):
        n = x.shape[1]
        self.model = nl.net.newff([[-1.0, 1.0] for _ in range(n)], [5, 1])

    def raw_predict(self, x):
        if self.model is None:
            self.reset(x)

        return self.model.sim(x.toarray())

    def predict(self, x):
        if self.raw_predict(x) > 0.0:
            return 1.0
        return -1.0

    def partial_fit(self, x, y, sample_weight=1.0):
        if self.model is None:
            self.reset(x)

        x = np.array(x.toarray())
        y = np.array([[y]])
        self.model.train(x, y, epochs=10, goal=0.0, show=False)
