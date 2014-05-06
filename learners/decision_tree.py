"""
    An incremental decision tree weak learner that operates in the style
    of Utgoff (i.e., by updating counts, splitting on best attribute,
    and propagating examples downward), but in a less efficient manner.
"""


import dtree


class DecisionTree(object):

    def __init__(self, classes):
        self.classes = classes
        self.model = None

    def initialize(self, X):
        n = X.shape[1]
        self.convert_data = lambda x, y: dtree.Data(
            [list(x.toarray()[0]) + [y]],
            order=map(str, range(n)) + ['cls'],
            types=[dtree.CON] * n + [dtree.DIS],
            modes=dict(cls=dtree.CLS))
        self.model = dtree.Tree(dtree.Data(
            [],
            order=map(str, range(n)) + ['cls'],
            types=[dtree.CON] * n + [dtree.DIS],
            modes=dict(cls=dtree.CLS)), auto_grow=True, splitting_n=1)
        self.model.set_missing_value_policy(dtree.USE_NEAREST)

    def partial_fit(self, X, y, sample_weight=1.0):
        if not self.model:
            self.initialize(X)

        data = self.convert_data(X, y)
        for row in data:
            self.model.train(row)

    def predict(self, X):
        if not self.model:
            return self.classes[0]

        X = X.toarray()[0]
        X = {str(i): X[i] for i in range(len(X))}
        return self.model.predict(X).best
