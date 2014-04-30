"""
    An online boosting algorithm which mixes SmoothBoost with
    Online Convex Programming from Chen '12.
"""


def project(v, z=1.0):
    U = set(range(len(v)))
    s = 0.0
    rho = 0.0

    while U:
        k = U.pop()

        G = set([j for j in U if v[j] >= v[k]])
        G.add(k)
        L = set([j for j in U if v[j] < v[k]])

        delta_rho = len(G)
        delta_s = sum(v[j] for j in G)

        if (s + delta_s) - (rho + delta_rho) * v[k] < z:
            s += delta_s
            rho += delta_rho
            U = L
        else:
            G.remove(k)
            U = G
    theta = (s - z) / rho
    return [max(vi - theta, 0) for vi in v]


class OCPBooster(object):
    delta = 0.5
    gamma = 0.1
    theta = gamma / (2 + gamma)
    eta = 1.0

    def __init__(self, Learner, classes, M=None):
        if not M:
            self.M = int(1.0 / (self.delta * self.gamma * self.gamma))
        else:
            self.M = M
        self.z = [0.0 for _ in range(self.M)]
        self.alpha = [1.0 / self.M for _ in range(self.M)]
        self.learners = [Learner(classes) for _ in range(self.M)]

    def update(self, features, label):

        def f(x):
            return sum(a * h.predict(x) for (a, h) in zip(self.alpha, self.learners))

        def normalize(x):
            norm = sum(x)
            return [xi / norm for xi in x]

        if label * f(features) < self.theta:
            for i in range(self.M):
                self.alpha[i] += self.eta * label * \
                    self.learners[i].predict(features)
                self.alpha = project(self.alpha)

        w = [1]
        for i in range(self.M):
            if i == 0:
                initial = 0
            else:
                initial = self.z[i]

            self.z[i] = initial + label * \
                self.learners[i].predict(features) - self.theta
            self.learners[i].partial_fit(
                features, label, sample_weight=w[i])
            w.append(min((1 - self.gamma) ** (self.z[i] / 2), 1))

    def predict(self, features):
        f = sum(self.alpha[i] * self.learners[i].predict(features)
                for i in range(self.M))
        if f >= 0:
            return 1
        else:
            return -1
