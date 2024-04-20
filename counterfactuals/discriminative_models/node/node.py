import torch
from torch import nn
from .odst_block import DenseBlock
from tqdm import tqdm
import numpy as np


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class NODE(nn.Module):
    def __init__(
        self,
        input_size,
        target_size,
        hidden_features=2048,
        num_layers=1,
        depth=6,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.dense_block = DenseBlock(
            input_dim=input_size,
            layer_dim=hidden_features,
            num_layers=num_layers,
            tree_dim=target_size,
            depth=depth,
            flatten_output=False,
        )
        self.output_layer = Lambda(lambda x: torch.mean(x, dim=1))
        self.target_size = target_size
        if target_size == 1:
            self.final_activation = torch.nn.Sigmoid()
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.prep_for_loss = lambda x: x.view(-1, 1).float()
        else:
            self.final_activation = torch.nn.Softmax(dim=1)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.prep_for_loss = lambda x: x.view(-1).long()

    def forward(self, x, y=None):
        x = self.dense_block(x)
        y = self.output_layer(x)
        return y

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
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
                if test_loss < (min_test_loss + eps):
                    min_test_loss = test_loss
                    patience_counter = 0
                    self.save(checkpoint_path)
                else:
                    patience_counter += 1
                if patience_counter > patience:
                    break
                self.load(checkpoint_path)
            pbar.set_description(
                f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
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
