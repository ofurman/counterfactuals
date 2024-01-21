import numpy as np


class BlackBox:

    def __init__(self, model):
        self.model = model
        if hasattr(self.model, 'predict_proba'):
            self.pred_fn = self.model.predict_proba
        else:
            self.pred_fn = self.model.predict

    def predict(self, X):

        proba = self.pred_fn(X)
        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = proba.flatten()
            classes = np.array([1 if y_pred > 0.5 else 0 for y_pred in classes])
        return classes

    def predict_proba(self, X):

        probs = self.pred_fn(X)
        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs
