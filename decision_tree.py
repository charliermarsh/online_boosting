"""
    An incremental decision tree based on Utgoff [1997].
"""
from math import log
from sys import maxint


def best_metric(examples):
    n = len(examples[0][0])
    num_examples = len(examples)
    best_thresh = None
    best_attribute = None
    minimum = maxint
    for i in range(n):
        def best_metric_for_attribute(i):
            best_thresh = None
            minimum = maxint

            left = sorted(examples, key=lambda x: x[0][i])
            right = []
            num_left = len(left)
            pos_left = len(filter(lambda x: x[1] == 1, examples))
            num_right = 0
            pos_right = 0

            while left:
                ex = None
                while not ex or (left and left[-1][0][i] == ex[0][i]):
                    ex = left.pop()
                    right.append(ex)
                    if ex[1] == 1:
                        pos_left -= 1
                        pos_right += 1
                    num_left -= 1
                    num_right += 1

                def entropy(p, n, N):
                    if n == 0:
                        return maxint
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

            return minimum, best_thresh

        score, thresh = best_metric_for_attribute(i)
        if score < minimum:
            minimum = score
            best_thresh = thresh
            best_attribute = i

    return (lambda x: x[best_attribute] >= best_thresh, best_attribute, best_thresh)


class DecisionTree(object):

    def __init__(self):
        self.root = Node()

    def predict(self, x):
        return self.root.predict(x)

    def update(self, x, y):
        incremental_update(self.root, (x, y))


class Node(object):

    def __init__(self):
        self.label = None
        self.examples = []
        self.left = None
        self.right = None
        self.decision = False
        self.test = None

        self.attr = None
        self.thresh = None

    def predict(self, x):
        if self.decision:
            if self.test(x):
                return self.left.predict(x)
            else:
                return self.right.predict(x)
        else:
            return self.label

    def display(self):
        if self.decision:
            print "Attribute = " + str(self.attr) + ", Threshold = " + str(self.thresh)
        else:
            print "Leaf node with label " + str(self.label)
        if self.left:
            self.left.display()
        if self.right:
            self.right.display()


def incremental_update(node, example):
    add_example_to_tree(node, example)
    ensure_best_test(node)


def ensure_best_test(node):
    if node is None:
        return

    if node.decision:
        (best_test, best_attribute, best_thresh) = best_metric(node.examples)
        if best_attribute != node.attr and best_thresh != node.thresh:
            node.test = best_test
            node.attr = best_attribute
            node.thresh = best_thresh
            node.left = Node()
            node.right = Node()
            for (x, y) in node.examples:
                if node.test(x):
                    add_example_to_tree(node.left, (x, y))
                else:
                    add_example_to_tree(node.right, (x, y))
        if node.decision:
            ensure_best_test(node.right)
            ensure_best_test(node.left)


def add_example_to_tree(node, (example, label)):
    if node.decision:
        if node.test(example):
            add_example_to_tree(node.left, (example, label))
        else:
            add_example_to_tree(node.right, (example, label))
    else:
        if node.label is None:
            node.label = label

        node.examples.append((example, label))

        if node.label != label:
            node.decision = True

            node.left = Node()
            node.right = Node()

            (node.test, best_attribute,
             best_thresh) = best_metric(node.examples)
            node.attr = best_attribute
            node.thresh = best_thresh
            for (x, y) in node.examples:
                if node.test(x):
                    add_example_to_tree(node.left, (x, y))
                else:
                    add_example_to_tree(node.right, (x, y))


if __name__ == "__main__":
    from random import randint

    def random_example(k):
        return ([randint(1, 5) for i in range(k)], (-1) ** randint(0, 1))

    n = 5
    k = 3
    examples = [random_example(k) for i in range(n)]
    print examples

    tree = DecisionTree()
    for (features, label) in examples:
        tree.update(features, label)

    tree.root.display()

    for (features, label) in examples:
        if tree.predict(features) != label:
            print(features, label)
