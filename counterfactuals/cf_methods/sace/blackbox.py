import numpy as np


class BlackBox:
    def __init__(self, model):
        self.model = model
        # self.pred_fn = self.model.predict
        # if hasattr(self.model, 'predict_proba'):
        self.pred_fn = self.model.predict_proba
        # else:
        #     self.pred_fn = self.model.predict

    def predict(self, X):
        proba = self.pred_fn(X).numpy()
        if proba.shape[1] == 1:
            classes = np.array([1 if y_pred > 0.5 else 0 for y_pred in proba])
        else:
            classes = np.argmax(proba, axis=1)
        return classes

    def predict_proba(self, X):
        probs = self.pred_fn(X).numpy()
        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs
