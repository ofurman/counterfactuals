import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from counterfactuals.discriminative_models.base_predictor import BasePredictor


class MultilayerPerceptron(BasePredictor):

    def __init__(self, D, weights_path, HC=30):
        #print("Input: ", D)
        super(MultilayerPerceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(D, HC),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(HC, HC),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(HC, 2),
            #nn.Softmax(-1)
        )

        if weights_path != "":
            self.load(weights_path)

    def eval(self):
        self.layers.eval()

    def forward(self, x) -> torch.Tensor:
        return self.layers(x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_indices = list(range(X_train.size(0)))
        val_indices = list(range(X_val.size(0)))
        train_loader = DataLoader(train_indices, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_indices, batch_size=100, shuffle=False)

        epochs = 100
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=1e-3)

        prev_acc = 0
        best_state_dict = None
        device = "cpu"
        for epoch in range(epochs):
            train_loss, train_acc, val_acc = 0, 0, 0
            self.layers.train()
            for idx in tqdm(train_loader):
                x = X_train[idx, :].to(device)
                y = y_train[idx].to(device)
                yhat = self.forward(x)
                label = yhat.argmax(-1)
                loss = nn.CrossEntropyLoss()(yhat, y)
                acc = (label == y).sum(0) / len(idx)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                train_loss += loss.item()
                train_acc += acc.item()

            self.layers.eval()
            for idx in tqdm(val_loader):
                x = X_val[idx, :].to(device)
                y = y_val[idx].to(device)
                yhat = self.forward(x)
                label = yhat.argmax(-1)
                acc = (label == y).sum(0) / len(idx)
                val_acc += acc.item()

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)
            val_acc = val_acc / len(val_loader)

            msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train_acc: {train_acc:.3f} // Val_acc: {val_acc:.3f}"
            print(msg)

            if val_acc > prev_acc:
                best_state_dict = self.layers.state_dict()
                prev_acc = val_acc

        self.layers.load_state_dict(best_state_dict)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)
        probs = np.argmax(probs, axis=1)
        return probs

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        with torch.no_grad():
            logits = self.forward(X_test)
            probs = torch.softmax(logits, dim=-1)
            #if self.target_size == 1:
            #    probs = torch.hstack([1 - probs, probs])
            return probs.float().numpy()

    def save(self, path: str):
        torch.save(self.layers.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path)
        self.layers.load_state_dict(state_dict)

