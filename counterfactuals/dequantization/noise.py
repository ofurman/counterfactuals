from typing import Callable, Dict

import numpy as np


def uniform_noise(rng, shape):
    return rng.rand(*shape)


def gaussian_noise(rng, shape):
    z = rng.randn(*shape)
    return sigmoid(z)


def logistic_noise(rng, shape):
    u = rng.rand(*shape)
    z = np.log(u) - np.log1p(-u)
    return sigmoid(z)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


NOISE_REGISTRY: Dict[str, Callable] = {
    "uniform": uniform_noise,
    "gaussian": gaussian_noise,
    "logistic": logistic_noise,
}
