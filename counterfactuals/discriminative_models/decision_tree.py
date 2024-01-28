import torch
import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier as DTClassifier

class DecisionTreeClassifier():    
    # build the constructor
    def __init__(self, max_depth=7, random_state=42, **decision_tree_kwargs):
        self.model = DTClassifier(max_depth=max_depth, random_state=random_state, **decision_tree_kwargs)

    def to(self, device):
        return self
    
    def eval(self):
        return self
    
    def parameters(self):
        return []
    
    def fit(self, train_loader, epochs=200, lr=0.003):
        X_train, y_train = train_loader.dataset.tensors
        X_train = X_train.numpy()
        y_train = y_train.numpy()
        self.model.fit(X_train, y_train)

    def forward(self, x):
        with torch.no_grad():
            y_pred = self.model.predict_proba(x)[:, 1]
        return torch.from_numpy(y_pred.reshape(-1, 1))

    def predict(self, X_test):
        with torch.no_grad():
            probs = self.forward(torch.from_numpy(X_test))
            probs = probs > 0.5
            return np.squeeze(probs.numpy().astype(np.float32))
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
