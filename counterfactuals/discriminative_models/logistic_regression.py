import numpy as np
import torch
from tqdm import tqdm

from counterfactuals.discriminative_models.base import BaseDiscModel


class LogisticRegression(BaseDiscModel):
    def __init__(self, input_size, target_size, device="auto"):
        super(LogisticRegression, self).__init__()
        # Auto-detect device if not specified or use CUDA if available
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.input_size = input_size
        self.target_size = target_size
        self.linear = torch.nn.Linear(input_size, target_size)

        # Move model to specified device
        self.to(self.device)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
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
        criterion = torch.nn.BCELoss()
        patience_counter = 0
        min_test_loss = float("inf")
        self.train()
        for epoch in (pbar := tqdm(range(epochs))):
            train_loss = 0.0
            for i, (examples, labels) in enumerate(train_loader):
                # Move data to device
                examples = examples.to(self.device)
                labels = labels.to(self.device)

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
                        # Move data to device
                        examples = examples.to(self.device)
                        labels = labels.to(self.device)

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

    def predict(self, X_test):
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).type(torch.float32)

        # Move input to device
        X_test = X_test.to(self.device)

        self.eval()
        with torch.no_grad():
            probs = self.forward(X_test)
            probs = probs > 0.5
            # Move result back to CPU for compatibility with sklearn and numpy
            return probs.float().view(-1).cpu()

    def predict_proba(self, X_test):
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).type(torch.float32)

        # Move input to device
        X_test = X_test.to(self.device)

        self.eval()
        with torch.no_grad():
            probs = self.forward(X_test).type(torch.float32)
            probs = torch.hstack([1 - probs, probs]).detach().float()
            # Move result back to CPU for compatibility with sklearn and numpy
            return probs.cpu()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        # Load with proper device mapping
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)


class MultinomialLogisticRegression(BaseDiscModel):
    def __init__(self, input_size, target_size, device="auto"):
        super(MultinomialLogisticRegression, self).__init__()
        # Auto-detect device if not specified or use CUDA if available
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.linear = torch.nn.Linear(input_size, target_size)

        # Move model to specified device
        self.to(self.device)

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
                # Move data to device
                examples = examples.to(self.device)
                labels = labels.to(self.device)

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
                        # Move data to device
                        examples = examples.to(self.device)
                        labels = labels.to(self.device)

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

    def predict(self, X_test: np.ndarray):
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).type(torch.float32)

        # Move input to device
        X_test = X_test.to(self.device)

        self.eval()
        with torch.no_grad():
            probs = self(X_test)
            predicted = torch.argmax(probs, 1)
            # Move result back to CPU for compatibility with sklearn and numpy
            return predicted.float().cpu()

    def predict_proba(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).type(torch.float32)

        # Move input to device
        X_test = X_test.to(self.device)

        self.eval()
        with torch.no_grad():
            probs = self.forward(X_test)
            probs = torch.nn.functional.softmax(probs, dim=1)
            # Move result back to CPU for compatibility with sklearn and numpy
            return probs.float().cpu()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        # Load with proper device mapping
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
