from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from counterfactuals.models.classifier_mixin import ClassifierPytorchMixin
from counterfactuals.models.pytorch_base import PytorchBase

from .odst_block import DenseBlock


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class NODE(PytorchBase, ClassifierPytorchMixin):
    def __init__(
        self,
        num_inputs: int,
        num_targets: int,
        hidden_features: int = 2048,
        num_layers: int = 1,
        depth: int = 6,
        device: str = "cpu",
    ):
        super().__init__(num_inputs, num_targets)
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.depth = depth
        self.device = device

        self.dense_block = DenseBlock(
            input_dim=num_inputs,
            layer_dim=hidden_features,
            num_layers=num_layers,
            tree_dim=num_targets,
            depth=depth,
            flatten_output=False,
        )
        self.output_layer = Lambda(lambda x: torch.mean(x, dim=1))

        if num_targets == 1:
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
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        lr: float = 0.001,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
        **kwargs,
    ) -> None:
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
