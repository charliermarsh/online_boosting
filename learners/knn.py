from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class kNN(object):

    def __init__(self, classes):
        self.model = KNeighborsClassifier(n_neighbors=2)
        self.X = None
        self.y = None

    def partial_fit(self, x, y):
        if self.X is None and self.y is None:
            self.X = np.array([x])
            self.y = y
        else:
            self.X = np.vstack((self.X, x))
            self.y = np.hstack((self.y, y))

        self.model.fit(self.X, self.y)

    def predict(self, x):
        return self.model.predict(x)[0]
