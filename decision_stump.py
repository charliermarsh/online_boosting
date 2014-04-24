"""
    An incremental decision tree based on Utgoff [1997].
"""
from math import log


def best_metric(examples):
    n = len(examples[0][0])
    num_examples = len(examples)
    best_thresh = None
    best_attribute = None
    minimum = 1
    for i in range(n):
        def best_metric_for_attribute(i):
            best_thresh = None
            minimum = 1

            left = sorted(examples, key=lambda x: x[0][i])
            right = []
            num_left = len(left)
            pos_left = len(filter(lambda x: x[1] == 1, examples))
            num_right = 0
            pos_right = 0

            for j in range(len(examples)):
                ex = left.pop()
                right.append(ex)
                if ex[1] == 1:
                    pos_left -= 1
                    pos_right += 1
                num_left -= 1
                num_right += 1

                def entropy(p, n, N):
                    if n == 0:
                        return 0
                    x = float(p) / n
                    y = float(n - p) / n
                    if x > 0:
                        x = x * log(x)
                    if y > 0:
                        y = y * log(y)
                    return -float(n) / N * (x + y)

                score = entropy(pos_left, num_left, num_examples) + \
                    entropy(pos_right, num_right, num_examples)

                if score < minimum:
                    best_thresh = ex[0][i]
                    minimum = score
            return score, best_thresh

        score, thresh = best_metric_for_attribute(i)
        if score < minimum:
            minimum = score
            best_thresh = thresh
            best_attribute = i

    return (best_attribute, best_thresh)


class DecisionStump(object):

    def __init__(self):
        self.examples = []
        self.comparator = None

    def predict(self, x):
        if self.comparator(x):
            return 1
        else:
            return -1

    def update(self, x, y):
        self.examples.append((x, y))
        (attribute, threshold) = best_metric(self.examples)

        right_examples = filter(
            lambda x: x[0][attribute] >= threshold, self.examples)
        pos = len(filter(lambda x: x[1] == 1, right_examples))
        neg = len(filter(lambda x: x[1] == -1, right_examples))
        if pos >= neg:
            self.comparator = lambda x: x[attribute] >= threshold
        else:
            self.comparator = lambda x: x[attribute] < threshold

if __name__ == "__main__":
    from random import randint

    def random_example(k):
        return ([randint(1, 50) for i in range(k)], (-1) ** randint(0, 1))

    n = 5
    k = 3
    examples = [random_example(k) for i in range(n)]

    stump = DecisionStump()
    for (features, label) in [random_example(k) for i in range(n)]:
        stump.update(features, label)
    print best_metric(stump.examples)
