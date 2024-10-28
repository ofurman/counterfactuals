import numpy as np
import torch
from tqdm import tqdm
from counterfactuals.discriminative_models.base import BaseDiscModel


class LinearRegression(BaseDiscModel):
    def __init__(self, input_size, target_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, target_size)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def fit(self, train_loader, test_loader=None, epochs=200, lr=0.003, patience=100):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        patience_counter = 0
        for epoch in (pbar := tqdm(range(epochs))):
            losses = []
            test_losses = []
            min_loss = float("inf")
            for i, (examples, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(examples)
                labels = labels.reshape(-1, 1)
                loss = criterion(outputs, labels.view(outputs.shape).float())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            if test_loader:
                with torch.no_grad():
                    for i, (examples, labels) in enumerate(test_loader):
                        outputs = self.forward(examples)
                        labels = labels.reshape(-1, 1)
                        test_loss = criterion(
                            outputs, labels.view(outputs.shape).float()
                        )
                        test_losses.append(test_loss.item())
                if np.mean(test_losses) < min_loss:
                    min_loss = np.mean(test_losses)
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter == patience:
                    break

            pbar.set_description(
                f"Epoch {epoch}, Train Loss: {np.mean(losses):.4f}, Test Loss: {np.mean(test_losses):.4f}, Patience: {patience_counter}"
            )

    def predict(self, X_test):
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).type(torch.float32)
        with torch.no_grad():
            preds = self.forward(X_test)
            return preds.float()

    def predict_proba(self, X_test):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
