from random import shuffle
from collections import defaultdict
import numpy as np


def test(Booster, Learner, data, m, trials=1, should_shuffle=True):
    results = []
    for t in range(trials):
        if should_shuffle:
            shuffle(data)
        results.append(run_test(Booster, Learner, data, m))
    results = zip(*results)

    def avg(x):
        return sum(x) / len(x)
    return (map(avg, zip(*results[0])), map(avg, zip(*results[1])))


def run_test(Booster, Learner, data, m):
    classes = np.unique(np.array([y for (x, y) in data]))
    baseline = Learner(classes)
    predictor = Booster(Learner, classes=classes, M=m)
    correct_booster = 0.0
    correct_baseline = 0.0
    t = 0
    performance_booster = []
    performance_baseline = []
    for (features, label) in data:
        if predictor.predict(features) == label:
            correct_booster += 1
        predictor.update(features, label)
        if baseline.predict(features) == label:
            correct_baseline += 1
        baseline.partial_fit(features, label)
        t += 1
        performance_booster.append(correct_booster / t)
        performance_baseline.append(correct_baseline / t)

    return performance_booster, performance_baseline


def testNumLearners(Booster, Learner, data, start, end, inc, trials=1):
    results = defaultdict(int)
    for t in range(trials):
        shuffle(data)
        for m in range(start, end + 1, inc):
            accuracy = test(Booster, Learner, data, m)[0][-1]
            print m, accuracy
            results[m] += accuracy
    for m in results:
        results[m] /= trials
    return results
