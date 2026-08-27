"""Shared selection of the factual instances a run explains.

Every counterfactual pipeline has to answer the same two questions before it
generates anything: which test rows get explained, and what target class each of
them is pushed towards. Keeping the answer here means DiCE, CCHVAE and DiCoFlex
select identical query sets for identical settings, which is what makes their
metrics comparable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def select_factual_indices(
    y_test: np.ndarray,
    target_class: Optional[int],
    n_test_samples: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Choose which test rows to explain.

    Args:
        y_test: Test labels, already relabelled by the discriminator if the
            experiment does that.
        target_class: The class counterfactuals are pushed towards. Rows already
            in that class are dropped, since flipping them is a no-op. When None,
            no row is dropped and both flip directions are covered in one run.
        n_test_samples: Optional cap on the number of explained rows, drawn
            without replacement. None explains every eligible row.
        seed: Seed for the subsample. A dedicated generator is used rather than
            the global NumPy state, so the query set depends only on this seed
            and not on how much randomness the pipeline consumed beforehand.

    Returns:
        Sorted array of indices into `y_test`.
    """
    if target_class is None:
        indices = np.arange(len(y_test))
    else:
        indices = np.flatnonzero(y_test != target_class)

    if n_test_samples is not None and 0 < n_test_samples < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=n_test_samples, replace=False))

    logger.info(
        "Explaining %d test rows (target_class=%s, n_test_samples=%s)",
        len(indices),
        target_class,
        n_test_samples,
    )
    return indices


def resolve_target_labels(y_factual: np.ndarray, target_class: Optional[int]) -> np.ndarray:
    """Return the desired class for each factual.

    Args:
        y_factual: Labels of the selected factuals.
        target_class: Fixed target class, or None to flip each factual's own
            label (binary tasks only).

    Returns:
        Array of target labels aligned with `y_factual`.
    """
    if target_class is None:
        return np.abs(1 - y_factual)
    return np.full_like(y_factual, fill_value=target_class)
