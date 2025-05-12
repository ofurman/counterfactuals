import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# progress bar
from tqdm import tqdm


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# a simple pytorch nn sequential model wrapper
class NNModel:
    def __init__(self, input_size, hidden_sizes, output_size):
        layers = []
        prev_size = input_size
        for curr_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, curr_size))
            layers.append(nn.ReLU())
            prev_size = curr_size
        layers.append(nn.Linear(prev_size, output_size))
        # no ReLU on the output
        self.layers = layers

        self.model = nn.Sequential(*layers)
        if output_size == 1:  # it is a binary classification
            self.loss_f = nn.BCEWithLogitsLoss()
        else:
            self.loss_f = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters())

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            res = self.model(x)
        return res.numpy()

    def train(self, X_train, y_train, epochs=50, batch_size=64):
        dataset = SimpleDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("Training:")
        self.model.train()
        for _ in tqdm(range(epochs)):
            for _, (X, y) in enumerate(dataloader):
                y_pred = self.model(X)

                loss = self.loss_f(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def test(self, X_test, y_test):
        dataset = SimpleDataset(X_test, y_test)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        print("Testing:")
        self.model.eval()
        losses = []
        correct = []
        with torch.no_grad():
            for _, (X, y) in enumerate(dataloader):
                y_pred = self.model(X)
                losses.append(self.loss_f(y_pred, y).item())
                if y_pred.shape[1] > 1:  # multi class
                    class_pred = torch.argmax(y_pred, dim=1)
                    correct.append((class_pred == y).item())
                else:
                    correct.append(((y_pred > 0) == y).item())
        print(f"Accuracy: {sum(correct) / y_test.shape[0] * 100:.2f}%")
        print("Average loss:", sum(losses) / y_test.shape[0])

    def save(self, path="model.pt"):
        torch.save([self.model.state_dict(), self.layers], path)

    def save_onnx(self, path="model.onnx"):
        input_size = self.model[0].in_features
        # print("input", input_size)
        nn_input = torch.randn(1, input_size)
        torch.onnx.export(
            self.model,
            nn_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def load(self, path="model.pt"):
        state_dict, self.layers = torch.load(path)
        self.model = nn.Sequential(*self.layers)
        self.model.load_state_dict(state_dict)
