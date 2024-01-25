import torch
import numpy as np
from tqdm import tqdm

class MultilayerPerceptron(torch.nn.Module):    
    def __init__(self, layer_sizes):
        super(MultilayerPerceptron, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.relu(self.layers[i](x))
        x = torch.sigmoid(x)
        return x
    
    def fit(self, train_loader, epochs=200):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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
