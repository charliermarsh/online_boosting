from sklearn.datasets import load_svmlight_file
from adaboost import AdaBooster


def loadData(filename):
    X, y = load_svmlight_file(filename)

    def format((x, y)):
        return (list(x.data), y)
    return [format(example) for example in zip(X, y)]


if __name__ == "__main__":
    data = loadData("data/heart.txt")

    # Begin online learning
    booster = AdaBooster()
    correct = 0.0
    t = 0
    performance = []
    for (features, label) in data:
        if booster.predict(features) == label:
            correct += 1
        booster.update(features, label)
        t += 1
        performance.append(correct / t)

    print performance
