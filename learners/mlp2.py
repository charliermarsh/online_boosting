from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
import numpy as np

class MLP(object):

    def __init__(self, classes):
        if set(classes) != set([-1.0, 1.0]):
            raise ValueError

        self.fnn = None
        self.classes = classes
        self.num_classes = len(classes)
        self.features = 0

    def reset(self, x):
        self.num_features = x.shape[1]
        self.fnn = buildNetwork(
            self.num_features, 5, self.num_classes, outclass=SoftmaxLayer)

    def raw_predict(self, x):
        if self.fnn is None:
            self.reset(x)

        predictions = self.fnn.activate(x.toarray()[0])
        return -predictions[0] + predictions[1]

    def predict(self, x):
        if self.fnn is None:
            self.reset(x)

        if self.raw_predict(x) > 0.0:
            return 1.0
        return -1.0

    def partial_fit(self, x, y, sample_weight=1.0):
        if self.fnn is None:
            self.reset(x)

        if y == -1:
            y = 0

        temp_data = ClassificationDataSet(
            self.num_features, 1, nb_classes=self.num_classes)
        temp_data.addSample(x.toarray()[0], y)
        temp_data._convertToOneOfMany()
        trainer = BackpropTrainer(
            self.fnn, dataset=temp_data, momentum=0.1, weightdecay=0.01)
        trainer.trainEpochs(10)
