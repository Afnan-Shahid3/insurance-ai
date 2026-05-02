import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilityThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper classifier that applies a tuned probability threshold to predict()"""

    def __init__(self, base_classifier=None, threshold=0.5):
        self.base_classifier = base_classifier
        self.threshold = float(threshold)

    def fit(self, X, y):
        self.base_classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def __getattr__(self, name):
        return getattr(self.base_classifier, name)
