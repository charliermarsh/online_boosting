from random import shuffle
import warnings

from sklearn.datasets import load_svmlight_file
from utils.experiment import test

from ensemblers.adaboost import AdaBooster
from ensemblers.ogboost import OGBooster
from ensemblers.ocpboost import OCPBooster
from ensemblers.expboost import EXPBooster

from learners.sk_naive_bayes import NaiveBayes
from learners.perceptron import Perceptron
from learners.random_stump import RandomStump
from learners.decision_stump import DecisionStump
from learners.decision_tree import DecisionTree
from learners.knn import kNN
from learners.histogram import RNB
from learners.winnow import Winnow

warnings.filterwarnings("ignore", module="sklearn")


def loadData(filename):
    X, y = load_svmlight_file(filename)

    data = zip(X, y)
    shuffle(data)
    return data

if __name__ == "__main__":
    data = loadData("data/heart.txt")
    booster, baseline = test(AdaBooster, DecisionTree, data, 50)
    print booster[-1], baseline[-1]
