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
                loss = criterion(outputs, labels)
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
