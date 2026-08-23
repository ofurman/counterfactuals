"""Global RNG seeding for reproducible pipeline runs."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed every RNG the pipelines draw from.

    Seeding torch alone is not sufficient. DiCE's ``random`` generation method
    and the group dequantizer both draw from the NumPy global RNG, and several
    dataset helpers use the stdlib ``random`` module, so a run seeded only
    through ``torch.manual_seed`` varies uncontrollably between repetitions.

    Args:
        seed: Seed applied to ``random``, NumPy and torch.
        deterministic_torch: Also force deterministic cuDNN kernels. Slower,
            and unnecessary while the traintest pipelines pin themselves to CPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %s", seed)
