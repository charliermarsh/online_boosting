from sklearn.tree import DecisionTreeClassifier
import numpy as np


class DecisionTree(object):

    def __init__(self, classes):
        self.model = DecisionTreeClassifier()
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
