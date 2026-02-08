from typing import Callable

from cel.metrics.base import Metric

_METRIC_REGISTRY: dict[str, type[Metric]] = {}


def register_metric(key: str) -> Callable[[type[Metric]], type[Metric]]:
    """Decorator to register a metric class under a config 'name' key."""

    def _wrap(cls: type[Metric]) -> type[Metric]:
        if key in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{key}' already registered.")
        _METRIC_REGISTRY[key] = cls
        return cls

    return _wrap
