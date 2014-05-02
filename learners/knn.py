from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class kNN(object):

    def __init__(self, classes):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.X = None
        self.y = None

    def partial_fit(self, x, y, sample_weight=1.0):
        if self.X is None and self.y is None:
            self.X = x.toarray()
            self.y = np.array([y])
        else:
            self.X = np.vstack((self.X, x.toarray()))
            self.y = np.hstack((self.y, y))

        self.model.fit(self.X, self.y)

    def predict(self, x):
        if self.X is None or self.y is None:
            return 0

        return self.model.predict(x.toarray())[0]
