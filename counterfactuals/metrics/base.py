from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Metric(Protocol):
    name: str

    def required_inputs(self) -> set[str]: ...

    def __call__(self, **inputs: Any) -> Any: ...
