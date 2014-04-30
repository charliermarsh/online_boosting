from sklearn.naive_bayes import MultinomialNB
from sk_wrapper import Wrapper


class NaiveBayes(Wrapper):

    def __init__(self, classes):
        self.model = MultinomialNB()
        self.classes = classes
