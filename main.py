from sklearn.datasets import load_svmlight_file
from adaboost import AdaBooster


def loadData(filename):
    X, y = load_svmlight_file(filename)

    def format((x, y)):
        return (list(x.data), y)
    return [format(example) for example in zip(X, y)]


if __name__ == "__main__":
    # Prepare data
    train = 200
    test = 50
    data = loadData("data/heart.txt")
    trainSet = data[:train]
    testSet = data[train: train + test]

    # Begin online learning
    booster = AdaBooster()
    for (features, label) in trainSet:
        booster.update(features, label)

    error = 0
    for (features, label) in testSet:
        prediction = booster.predict(features)
        if prediction != label:
            error += 1

    print "Error rate: " + str(float(error) / test)
