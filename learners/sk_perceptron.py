from sklearn import linear_model
from sk_wrapper import Wrapper


class Perceptron(Wrapper):

    def __init__(self, classes):
        self.model = linear_model.Perceptron()
        self.classes = classes
