from sklearn.datasets import load_svmlight_file
from random import shuffle
from adaboost2 import AdaBooster
from ogboost import OGBooster
from ocpbooster import OCPBooster
from expbooster import EXPBooster
from naive_bayes import NaiveBayes as Learner1
from perceptron import Perceptron as Learner2
from random_stump import RandomStump as Learner3
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
    data = loadData("data/heart.txt")
    print test(EXPBooster, Learner3, data, 100)[-1]
