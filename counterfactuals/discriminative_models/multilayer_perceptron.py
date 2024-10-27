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
        self.input_size = input_size
        layer_sizes = [input_size] + hidden_layer_sizes + [target_size]
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.target_size = target_size
        if target_size == 1:
            self.final_activation = torch.nn.Sigmoid()
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.prep_for_loss = lambda x: x.view(-1, 1).float()
        else:
            self.final_activation = torch.nn.Softmax(dim=1)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.prep_for_loss = lambda x: x.view(-1).long()

    def forward(self, x):
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                x = self.layers[i](x)
            else:
                x = self.relu(self.dropout(self.layers[i](x)))
        return x

    def fit(
        self,
        train_loader,
        test_loader=None,
        epochs=200,
        lr=0.001,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
    ):
        min_test_loss = float("inf")
        optimizer = torch.optim.RAdam(self.parameters(), lr=lr)
        for epoch in (pbar := tqdm(range(epochs))):
            train_loss = 0
            test_loss = 0
            for i, (examples, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(examples)
                loss = self.criterion(outputs, self.prep_for_loss(labels))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)
            if test_loader:
                with torch.no_grad():
                    for i, (examples, labels) in enumerate(test_loader):
                        outputs = self.forward(examples)
                        loss = self.criterion(outputs, self.prep_for_loss(labels))
                        test_loss += loss.item()
                        # Early stopping
                    test_loss /= len(test_loader)
                if test_loss < (min_test_loss - eps):
                    min_test_loss = test_loss
                    patience_counter = 0
                    self.save(checkpoint_path)
                else:
                    patience_counter += 1
                if patience_counter > patience:
                    break
                self.load(checkpoint_path)
            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, test: {test_loss:.4f}, patience: {patience_counter}"
            )

    def predict(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        with torch.no_grad():
            probs = self.predict_proba(X_test)
            probs = torch.argmax(probs, dim=1)
            return probs.squeeze().float()

    def predict_proba(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        with torch.no_grad():
            logits = self.forward(X_test)
            probs = self.final_activation(logits)
            if self.target_size == 1:
                probs = torch.hstack([1 - probs, probs])
            return probs.float()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
