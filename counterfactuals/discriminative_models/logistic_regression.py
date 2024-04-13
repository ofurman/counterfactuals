import numpy as np
import torch
from tqdm import tqdm
from counterfactuals.discriminative_models.base import BaseDiscModel


class LogisticRegression(BaseDiscModel):
    def __init__(self, input_size, target_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, target_size)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    def fit(self, train_loader, epochs=200, lr=0.003):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        for epoch in (pbar := tqdm(range(epochs))):
            losses = []
            for i, (examples, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(examples)
                labels = labels.reshape(-1, 1)
                loss = criterion(outputs, labels.float())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            pbar.set_description(f"Epoch {epoch}, Loss: {np.mean(losses):.4f}")

    def predict(self, X_test):
        with torch.no_grad():
            probs = self.forward(torch.from_numpy(X_test))
            probs = probs > 0.5
            return np.squeeze(probs.numpy().astype(np.float32))

    def predict_proba(self, X_test):
        with torch.no_grad():
            probs = self.forward(torch.from_numpy(X_test).type(torch.float32))
            probs = torch.hstack([1 - probs, probs]).detach().numpy().astype(np.float32)
            return probs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MultinomialLogisticRegression(BaseDiscModel):
    def __init__(self, input_size, target_size):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, target_size)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def fit(self, train_loader, test_loader=None, epochs=200, lr=0.003):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in (pbar := tqdm(range(epochs))):
            losses = []
            test_losses = []
            for i, (examples, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(examples)
                labels = labels.reshape(-1).type(torch.int64)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            if test_loader:
                with torch.no_grad():
                    for i, (examples, labels) in enumerate(test_loader):
                        labels = labels.type(torch.int64)
                        outputs = self.forward(examples)
                        loss = criterion(outputs, labels)
                        test_losses.append(loss.item())

                        # Early stopping
                        if i > 10 and np.mean(test_losses[-10:]) > np.mean(
                            test_losses[-5:]
                        ):
                            break
            pbar.set_description(
                f"Epoch {epoch}, Train Loss: {np.mean(losses):.4f}, Test Loss: {np.mean(test_losses):.4f}"
            )

    def predict(self, X_test: np.ndarray):
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).type(torch.float32)
        with torch.no_grad():
            probs = self(X_test)
            _, predicted = torch.max(probs, 1)
            return predicted

    def predict_proba(self, X_test):
        with torch.no_grad():
            probs = self.forward(torch.from_numpy(X_test).type(torch.float32))
            probs = torch.nn.functional.softmax(probs, dim=1)
            return probs.numpy().astype(np.float32)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
