import unittest
from perceptron import Perceptron


class TestPerceptron(unittest.TestCase):

    def testSimple(self):
        x1 = [1, 2, 3]
        y1 = 1
        x2 = [1, 3, 5]
        y2 = -1
        x3 = [-1, -1, -1]
        y3 = 1
        x4 = [1, 2, 4]
        y4 = 1
        data = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        p = Perceptron()
        for (x, y) in data:
            p.update(x, y)
        self.assertEqual(p.w, [1.0, 1.0, 2.0])

if __name__ == '__main__':
    unittest.main()
