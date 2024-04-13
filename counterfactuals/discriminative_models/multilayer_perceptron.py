import numpy as np
import torch
from tqdm import tqdm
from typing import List
from counterfactuals.discriminative_models.base import BaseDiscModel


class MultilayerPerceptron(BaseDiscModel):
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: List[int],
        target_size: int,
        dropout: float = 0.2,
    ):
        super(MultilayerPerceptron, self).__init__()
        self.target_size = target_size
        layer_sizes = [input_size] + hidden_layer_sizes + [target_size]
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        if target_size == 1:
            self.final_activation = torch.nn.Sigmoid()
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.final_activation = torch.nn.Softmax(dim=1)
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                x = self.layers[i](x)
            else:
                x = self.relu(self.dropout(self.layers[i](x)))
        return x

    def fit(self, train_loader, test_loader=None, epochs=200, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in (pbar := tqdm(range(epochs))):
            losses = []
            test_losses = []
            for i, (examples, labels) in enumerate(train_loader):
                labels = labels.type(torch.int64)
                optimizer.zero_grad()
                outputs = self.forward(examples)
                labels = labels.reshape(-1).type(torch.int64)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            if test_loader:
                with torch.no_grad():
                    for i, (examples, labels) in enumerate(test_loader):
                        labels = labels.type(torch.int64)
                        outputs = self.forward(examples)
                        loss = self.criterion(outputs, labels)
                        test_losses.append(loss.item())
                        # Early stopping
            if epoch > 10 and np.mean(test_losses[-10:]) > np.mean(test_losses[-5:]):
                break
            pbar.set_description(
                f"Epoch {epoch}, Train Loss: {np.mean(losses):.4f}, Test Loss: {np.mean(test_losses):.4f}"
            )

    def predict(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test)
        with torch.no_grad():
            probs = self.predict_proba(X_test)
            if self.target_size > 1:
                probs = torch.argmax(probs, dim=1)
            else:
                probs = probs > 0.5
            return np.squeeze(probs)

    def predict_proba(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test)
        with torch.no_grad():
            logits = self.forward(X_test.type(torch.float32))
            probs = self.final_activation(logits)
            # probs = torch.hstack([1 - probs, probs]).detach().numpy().astype(np.float32)
            return probs.type(torch.float32)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
