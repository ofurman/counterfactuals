import torch
from nflows.transforms import Transform


class Dequantization(Transform):
    """Dequantization of discrete data.

    This transform is used to dequantize discrete data by adding noise to it.
    The noise is sampled from a uniform distribution on the interval [0, 1).
    """

    def __init__(self, alpha=1e-5, category_counts=None, category_indices=None):
        super().__init__()
        self.alpha = torch.Tensor([alpha])
        self.category_counts = category_counts
        self.category_indices = category_indices

    def forward(self, inputs, context=None):
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)
        inputs_cat = inputs[:, self.category_indices]
        inputs_cat, ldj = self.dequant(inputs_cat, ldj)
        inputs_cat, ldj = self.sigmoid(inputs_cat, ldj, reverse=True)
        inputs[:, self.category_indices] = inputs_cat
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)
        return inputs, ldj

    def inverse(self, inputs, context=None):
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)
        inputs_cat = inputs[:, self.category_indices]

        inputs_cat, ldj = self.sigmoid(inputs_cat, ldj, reverse=False)

        for i, count in enumerate(self.category_counts):
            inputs_cat[:, i] = inputs_cat[:, i] * count
            inputs_cat[:, i] = (
                torch.floor(inputs_cat[:, i])
                .clamp(min=0, max=count - 1)
                .to(torch.int32)
            )
        ldj += sum(
            torch.log(torch.Tensor([count])) * inputs_cat.shape[1]
            for count in self.category_counts
        )
        inputs[:, self.category_indices] = inputs_cat
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)
        return inputs, ldj

    def sigmoid(self, z, ldj, reverse=False):
        if not reverse:
            ldj += (-z - 2 * torch.nn.functional.softplus(-z)).sum(dim=1)
            z = torch.sigmoid(z)
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
            ldj -= torch.log(1 - self.alpha) * torch.prod(torch.Tensor(z.shape[1:]))
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha
            ldj += torch.log(1 - self.alpha) * torch.prod(torch.Tensor(z.shape[1:]))
            ldj += (-torch.log(z) - torch.log(1 - z)).sum(dim=1)
            z = torch.log(z) - torch.log(1 - z)
        return z, ldj

    def dequant(self, z, ldj):
        z = z.to(torch.float32)
        for i, count in enumerate(self.category_counts):
            z[:, i] = z[:, i] + torch.rand_like(z[:, i]).detach()
            z[:, i] = z[:, i] / count
        ldj -= sum(
            torch.log(torch.Tensor([count])) * z.shape[1]
            for count in self.category_counts
        )
        return z, ldj
