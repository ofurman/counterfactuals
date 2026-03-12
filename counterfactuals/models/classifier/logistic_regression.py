import torch
from tqdm import tqdm

from counterfactuals.models.classifier_mixin import ClassifierPytorchMixin
from counterfactuals.models.pytorch_base import PytorchBase


class LogisticRegression(PytorchBase, ClassifierPytorchMixin):
    def __init__(self, num_inputs: int, num_targets: int):
        super(LogisticRegression, self).__init__(num_inputs, num_targets)
        self.linear = torch.nn.Linear(num_inputs, num_targets)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        return self.linear(x)

    def fit(
        self,
        train_loader,
        test_loader=None,
        epochs=200,
        lr=0.003,
        patience=20,
        eps=1e-3,
        checkpoint_path="checkpoint.pth",
    ):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        patience_counter = 0
        min_test_loss = float("inf")
        self.train()
        for epoch in (pbar := tqdm(range(epochs))):
            train_loss = 0.0
            for i, (examples, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(examples)
                labels = labels.reshape(-1, 1)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            if test_loader:
                self.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for i, (examples, labels) in enumerate(test_loader):
                        outputs = self.forward(examples)
                        labels = labels.reshape(-1, 1)
                        loss = criterion(outputs, labels.float())
                        test_loss += loss
                test_loss /= len(test_loader)
            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, test: {test_loss:.4f}, patience: {patience_counter}"
            )
            if test_loss < (min_test_loss - eps):
                min_test_loss = test_loss
                patience_counter = 0
                self.save(checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter > patience:
                break
        self.load(checkpoint_path)


class MultinomialLogisticRegression(PytorchBase, ClassifierPytorchMixin):
    def __init__(self, num_inputs: int, num_targets: int):
        super(MultinomialLogisticRegression, self).__init__(num_inputs, num_targets)
        self.linear = torch.nn.Linear(num_inputs, num_targets)
        self.final_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def fit(
        self,
        train_loader,
        test_loader=None,
        epochs=200,
        lr=0.003,
        patience=20,
        eps=1e-3,
        checkpoint_path="checkpoint.pth",
    ):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        patience_counter = 0
        min_test_loss = float("inf")
        for epoch in (pbar := tqdm(range(epochs))):
            train_loss = 0.0
            for i, (examples, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(examples)
                labels = labels.reshape(-1).type(torch.int64)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            if test_loader:
                with torch.no_grad():
                    test_loss = 0.0
                    for i, (examples, labels) in enumerate(test_loader):
                        labels = labels.type(torch.int64)
                        outputs = self.forward(examples)
                        loss = criterion(outputs, labels)
                        test_loss += loss
                test_loss /= len(test_loader)
            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, test: {test_loss:.4f}, patience: {patience_counter}"
            )
            if test_loss < (min_test_loss - eps):
                min_test_loss = test_loss
                patience_counter = 0
                self.save(checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter > patience:
                break
        self.load(checkpoint_path)
