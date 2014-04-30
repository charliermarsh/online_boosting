import unittest
import sys
sys.path.append("../")
import numpy as np
from ..learners.knn import kNN


class TestLearners(unittest.TestCase):

    def setUp(self):
        x1 = [1, 2, 3]
        y1 = 1
        x2 = [1, 3, 5]
        y2 = 1
        x3 = [-1, -1, -1]
        y3 = -1
        x4 = [1, 2, 4]
        y4 = 1
        x5 = [-0.5, -0.5, 0]
        y5 = -1
        data = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
        self.data = [(np.array(x), np.array([y])) for (x, y) in data]
        self.labels = np.array([-1, 1])

    def testKNN(self):
        model = kNN(self.labels)
        for (x, y) in self.data:
            model.partial_fit(x, y)
        print model.predict(np.array([0, 0, 0]))


if __name__ == '__main__':
    unittest.main()
