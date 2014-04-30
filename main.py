from sklearn.datasets import load_svmlight_file
from random import shuffle
from adaboost2 import AdaBooster
from ogboost import OGBooster
from ocpbooster import OCPBooster
from expbooster import EXPBooster
from naive_bayes import NaiveBayes
from perceptron import Perceptron
from random_stump import RandomStump
from winnow import Winnow
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
    data = loadData("data/german.numer_scale.txt")
    print test(AdaBooster, Winnow, data, 100)[-1]
