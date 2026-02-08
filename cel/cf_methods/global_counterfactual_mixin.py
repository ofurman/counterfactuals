from abc import ABC


class GlobalCounterfactualMixin(ABC):
    """
    Mixin class for global counterfactual explanation methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize global counterfactual functionality."""
        super().__init__(*args, **kwargs)
        self._method_type: str = "global"
