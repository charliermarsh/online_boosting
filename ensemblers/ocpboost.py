"""
    An online boosting algorithm which mixes SmoothBoost with
    Online Convex Programming from Chen '12.
"""

from math import sqrt
import numpy as np
from osboost import OSBooster


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
    return np.array([max(vi - theta, 0) for vi in v])


class OCPBooster(OSBooster):

    def update(self, features, label):
        try:
            self.t += 1
        except:
            self.t = 1

        if self.raw_predict(features) * label < self.theta:
            eta = 1 / sqrt(self.t)
            predictions = np.array([learner.predict(features)
                                    for learner in self.learners])
            self.alpha += eta * label * predictions
            self.alpha = project(self.alpha)

        super(OCPBooster, self).update(features, label)
