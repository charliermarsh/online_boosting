from sklearn.datasets import load_svmlight_file
from adaboost import AdaBooster
from ogboost import OGBooster
from histogram import RNB as Learner
from experiment import test
import warnings
warnings.filterwarnings("ignore", module="sklearn")


def loadData(filename):
    X, y = load_svmlight_file(filename)

    def format((x, y)):
        return (list(x.data), y)
    return [format(example) for example in zip(X, y)]


if __name__ == "__main__":
    data = loadData("data/german.numer_scale.txt")[:200]
    print test(AdaBooster, Learner, data, 20)
