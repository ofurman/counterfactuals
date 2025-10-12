from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    name: str

    @abstractmethod
    def required_inputs(self) -> set[str]:
        """Return the set of required input keys.

        Returns:
            set[str]: The set of required input keys.
        """

    @abstractmethod
    def __call__(self, **inputs: Any) -> Any:
        pass
