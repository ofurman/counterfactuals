"""Implementations of various mixture models."""

import abc

import numpy as np
import torch
from torch import distributions, nn

from counterfactuals.generative_models.base import BaseGenModel


def _default_sample_fn(logits):
    return distributions.Bernoulli(logits=logits).sample()


def auto_reshape(fn):
    """Decorator which flattens image inputs and reshapes them before returning.

    This is used to enable non-convolutional models to transparently work on images.
    """

    def wrapped_fn(self, x, *args, **kwargs):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        y = fn(self, x, *args, **kwargs)
        return y.view(original_shape)

    return wrapped_fn


class GenerativeModel(abc.ABC, nn.Module):
    """Base class inherited by all generative models in pytorch-generative.

    Provides:
        * An abstract `sample()` method which is implemented by subclasses that support
          generating samples.
        * Variables `self._c, self._h, self._w` which store the shape of the (first)
          image Tensor the model was trained with. Note that `forward()` must have been
          called at least once and the input must be an image for these variables to be
          available.
        * A `device` property which returns the device of the model's parameters.
    """

    def __call__(self, x, *args, **kwargs):
        """Saves input tensor attributes so they can be accessed during sampling."""
        if getattr(self, "_c", None) is None and x.dim() == 4:
            _, c, h, w = x.shape
            self._create_shape_buffers(c, h, w)
        return super().__call__(x, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Registers dynamic buffers before loading the model state."""
        if "_c" in state_dict and not getattr(self, "_c", None):
            c, h, w = state_dict["_c"], state_dict["_h"], state_dict["_w"]
            self._create_shape_buffers(c, h, w)
        super().load_state_dict(state_dict, strict)

    def _create_shape_buffers(self, channels, height, width):
        channels = channels if torch.is_tensor(channels) else torch.tensor(channels)
        height = height if torch.is_tensor(height) else torch.tensor(height)
        width = width if torch.is_tensor(width) else torch.tensor(width)
        self.register_buffer("_c", channels)
        self.register_buffer("_h", height)
        self.register_buffer("_w", width)

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def sample(self, n_samples): ...


class Kernel(abc.ABC, nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bandwidth=1.0):
        """Initializes a new Kernel.

        Args:
            bandwidth: The kernel's (band)width.
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    @abc.abstractmethod
    def forward(self, test_Xs, train_Xs):
        """Computes log p(x) for each x in test_Xs given train_Xs."""

    @abc.abstractmethod
    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class ParzenWindowKernel(Kernel):
    """Implementation of the Parzen window kernel."""

    def forward(self, test_Xs, train_Xs):
        abs_diffs = torch.abs(self._diffs(test_Xs, train_Xs))
        dims = tuple(range(len(abs_diffs.shape))[2:])
        dim = np.prod(abs_diffs.shape[2:])
        inside = torch.sum(abs_diffs / self.bandwidth <= 0.5, dim=dims) == dim
        coef = 1 / self.bandwidth**dim
        return torch.log((coef * inside).mean(dim=1))

    @torch.no_grad()
    def sample(self, train_Xs):
        device = train_Xs.device
        noise = (torch.rand(train_Xs.shape, device=device) - 0.5) * self.bandwidth
        return train_Xs + noise


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        n, d = train_Xs.shape
        n, h = torch.tensor(n, dtype=torch.float32), torch.tensor(self.bandwidth)
        pi = torch.tensor(np.pi)

        Z = 0.5 * d * torch.log(2 * pi) + d * torch.log(h) + torch.log(n)
        diffs = self._diffs(test_Xs, train_Xs) / h
        log_exp = -0.5 * torch.norm(diffs, p=2, dim=-1) ** 2

        return torch.logsumexp(log_exp - Z, dim=-1)

    @torch.no_grad()
    def sample(self, train_Xs):
        device = train_Xs.device
        noise = torch.randn(train_Xs.shape, device=device) * self.bandwidth
        return train_Xs + noise


class KernelDensityEstimator(GenerativeModel):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel=None):
        """Initializes a new KernelDensityEstimator.

        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.kernel = kernel or GaussianKernel()
        self.train_Xs = nn.Parameter(train_Xs, requires_grad=False)
        assert len(self.train_Xs.shape) == 2, "Input cannot have more than two axes."

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an
    # iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    @torch.no_grad()
    def sample(self, n_samples):
        idxs = np.random.choice(range(len(self.train_Xs)), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])


class KDE(BaseGenModel):
    def __init__(self, bandwidth=0.1, **kwargs):  # Ignores kwargs!
        super(KDE, self).__init__()
        self.bandwidth = bandwidth
        self.models = nn.ModuleDict()

    def _context_to_key(self, context):
        return str(int(context))

    def _get_model_for_context(self, context):
        key = self._context_to_key(context)
        if key not in self.models:
            raise ValueError(f"Context {key} not found in the model.")
        return self.models[key]

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        for key in state_dict.keys():
            if key.startswith("models."):
                self.models[key.split(".")[1]] = KernelDensityEstimator(
                    state_dict[key], kernel=GaussianKernel(bandwidth=self.bandwidth)
                )
        return super().load_state_dict(state_dict, strict, assign)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        checkpoint_path: str = "best_model.pth",
        **kwargs,
    ):
        train_Xs, train_ys = train_loader.dataset.tensors
        train_ys = train_ys.view(-1)
        for y in train_ys.unique():
            idxs = train_ys == y
            self.models.update(
                {
                    self._context_to_key(y.item()): KernelDensityEstimator(
                        train_Xs[idxs], kernel=GaussianKernel(bandwidth=self.bandwidth)
                    )
                }
            )
        self.save(checkpoint_path)

        train_log_probs = self.predict_log_prob(train_loader)
        test_log_probs = self.predict_log_prob(test_loader)
        print(f"Train log-likelihood: {train_log_probs.float().mean()}")
        print(f"Test log-likelihood: {test_log_probs.float().mean()}")

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        preds = torch.zeros_like(context)
        for i in range(x.shape[0]):
            model = self._get_model_for_context(context[i].item())
            preds[i] = model(x[i].unsqueeze(0))
        return preds.view(-1)

    def predict_log_prob(self, dataloader: torch.utils.data.DataLoader):
        inputs, context = dataloader.dataset.tensors
        preds = self(inputs, context)
        preds = torch.zeros_like(context, dtype=torch.float32)
        for i in range(inputs.shape[0]):
            model = self._get_model_for_context(context[i].item())
            preds[i] = model(inputs[i].unsqueeze(0))
        return preds

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
