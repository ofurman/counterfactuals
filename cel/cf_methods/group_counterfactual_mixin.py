from abc import ABC


class GroupCounterfactualMixin(ABC):
    """
    Mixin class for group-based counterfactual explanation methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize group counterfactual functionality."""
        super().__init__(*args, **kwargs)
        self._method_type: str = "group"
        self.n_groups: int | None = kwargs.get("n_groups", None)
