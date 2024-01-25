import torch
import numpy as np
from tqdm import tqdm

class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
    def fit(self, train_loader, epochs=200):
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=0.003)
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
            probs = torch.hstack([1-probs, probs]).detach().numpy().astype(np.float32)
            return probs
