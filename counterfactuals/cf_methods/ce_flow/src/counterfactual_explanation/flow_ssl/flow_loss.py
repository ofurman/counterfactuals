import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class FlowLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, prior, k=256):
        super().__init__()
        self.k = k
        self.prior = prior

    def forward(self, z, sldj, y=None):
        z = z.reshape((z.shape[0], -1))
        if y is not None:
            prior_ll = self.prior.log_prob(z, y)
        else:
            prior_ll = self.prior.log_prob(z)
        corrected_prior_ll = prior_ll - np.log(self.k) * np.prod(z.size()[1:])
        # PAVEL: why the correction?

        ll = corrected_prior_ll + sldj
        nll = -ll.mean()

        return nll


class FlowSoftmaxCELoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, local_batch, positive=False):
        validity_loss = 0
        if positive:
            validity_loss += F.hinge_embedding_loss(F.sigmoid(local_batch[:, 1]) - F.sigmoid(
                local_batch[:, 0]), torch.tensor(-1), self.margin, reduction='mean')
        else:
            validity_loss += F.hinge_embedding_loss(F.sigmoid(local_batch[:, 0]) - F.sigmoid(
                local_batch[:, 1]), torch.tensor(-1), self.margin, reduction='mean')
        return validity_loss

class FlowCrossEntropyCELoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, local_batch, positive=False):
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.BCELoss()
        if positive:
            target = torch.zeros(local_batch.shape)
        else:
            target = torch.ones(local_batch.shape)
        return loss_fn(local_batch.reshape(-1), target.reshape(-1))
