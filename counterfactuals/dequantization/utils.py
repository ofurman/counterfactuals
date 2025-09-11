import torch
import torch.nn as nn


class DequantizationWrapper(nn.Module):
    def __init__(self, gen_model, dequantizer):
        super().__init__()
        self.gen_model = gen_model
        self.dequantizer = dequantizer

    def forward(self, X, y=None):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        X = self.dequantizer.transform(X)
        X = torch.from_numpy(X)

        return self.gen_model(X) if y is None else self.gen_model(X, y)
