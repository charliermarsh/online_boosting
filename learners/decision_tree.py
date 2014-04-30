import dtree


class DecisionTree(object):

    def __init__(self, classes):
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
            modes=dict(cls=dtree.CLS)), auto_grow=True, splitting_n=10)
        self.model.set_missing_value_policy(dtree.USE_NEAREST)

    def partial_fit(self, X, y):
        if not self.model:
            self.initialize(X)

        data = self.convert_data(X, y)
        for row in data:
            self.model.train(row)

    def predict(self, X):
        X = X.toarray()[0]
        X = {str(i): X[i] for i in range(len(X))}
        return self.model.predict(X).best
