
def test(Booster, Learner, data, m):
    predictor = Booster(Learner, M=m)
    correct = 0.0
    t = 0
    performance = []
    for (features, label) in data:
        if predictor.predict(features) == label:
            correct += 1
        predictor.update(features, label)
        t += 1
        performance.append(correct / t)

    return performance


def testNumLearners(Booster, Learner, data, start=1, end=100):
    return {m: test(Booster, Learner, data, m) for m in range(start, end + 1)}
