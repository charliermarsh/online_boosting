from sklearn.datasets import load_svmlight_file
from random import shuffle
from adaboost import AdaBooster
from ogboost import OGBooster
from perceptron import Perceptron as Learner
from experiment import test
import warnings
warnings.filterwarnings("ignore", module="sklearn")


def loadData(filename):
    X, y = load_svmlight_file(filename)

    def format((x, y)):
        return (x.toarray()[0], y)
    data = [format(example) for example in zip(X, y)]
    shuffle(data)
    return data


if __name__ == "__main__":
    data = loadData("data/australian.txt")
    print test(AdaBooster, Learner, data, 100)
