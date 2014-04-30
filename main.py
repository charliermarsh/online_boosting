from sklearn.datasets import load_svmlight_file
import numpy as np
from random import shuffle
from ensemblers.adaboost import AdaBooster
from ensemblers.ogboost import OGBooster
from ensemblers.ocpboost import OCPBooster
from ensemblers.expboost import EXPBooster
from learners.sk_naive_bayes import NaiveBayes
from learners.perceptron import Perceptron
from learners.random_stump import RandomStump
from learners.decision_stump import DecisionStump
from learners.ce_knn import kNN
from learners.histogram import RNB
from learners.winnow import Winnow
from utils.experiment import test
import warnings
warnings.filterwarnings("ignore", module="sklearn")


def loadData(filename):
    X, y = load_svmlight_file(filename)

    data = zip(X, y)
    shuffle(data)
    return data


if __name__ == "__main__":
    data = loadData("data/german.numer_scale.txt")
    print test(AdaBooster, Winnow, data, 20)[-1]
