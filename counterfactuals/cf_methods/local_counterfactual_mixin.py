from abc import ABC


class LocalCounterfactualMixin(ABC):
    """
    Mixin class for local counterfactual explanation methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize local counterfactual functionality."""
        super().__init__(*args, **kwargs)
        self._method_type: str = "local"
