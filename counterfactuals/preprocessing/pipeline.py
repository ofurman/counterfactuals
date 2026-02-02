from typing import Optional

from counterfactuals.preprocessing.base import PreprocessingContext, PreprocessingStep


class PreprocessingPipeline(PreprocessingStep):
    """Pipeline for chaining multiple preprocessing steps.

    Allows sequential application of preprocessing transformations while
    maintaining a consistent context-based interface.

    Attributes:
        steps: List of (name, step) tuples defining the pipeline.
    """

    def __init__(self, steps: list[tuple[str, PreprocessingStep]]):
        """Initialize the preprocessing pipeline.

        Args:
            steps: List of (name, step) tuples where step implements PreprocessingStep.
        """
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        """Validate that all steps implement PreprocessingStep."""
        for name, step in self.steps:
            if not isinstance(step, PreprocessingStep):
                raise TypeError(
                    f"Step '{name}' must implement PreprocessingStep interface. "
                    f"Got {type(step)} instead."
                )

    def fit(self, context: PreprocessingContext) -> "PreprocessingPipeline":
        """Fit all steps in the pipeline sequentially.

        Each step is fitted on the transformed output of the previous step.
        Note: This method does NOT return transformed data, only fits the steps.
        Use fit_transform() if you want to fit and get transformed data.

        Args:
            context: Preprocessing context with data and feature indices.

        Returns:
            Self for method chaining.
        """
        current_context = context
        for name, step in self.steps:
            step.fit(current_context)
            # Transform for next step's fit, but don't return transformed data
            current_context = step.transform(current_context)
        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Apply all steps in the pipeline sequentially.

        Args:
            context: Preprocessing context with data to transform.

        Returns:
            New context with transformed data.
        """
        current_context = context
        for name, step in self.steps:
            current_context = step.transform(current_context)
        return current_context

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Apply inverse transformations in reverse order.

        Args:
            context: Preprocessing context with transformed data.

        Returns:
            New context with inverse transformed data.
        """
        current_context = context
        # Apply inverse transformations in reverse order
        for name, step in reversed(self.steps):
            current_context = step.inverse_transform(current_context)
        return current_context

    def fit_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Fit the pipeline and transform the data.

        Args:
            context: Preprocessing context with data and feature indices.

        Returns:
            New context with transformed data.
        """
        return self.fit(context).transform(context)

    def get_step(self, name: str) -> Optional[PreprocessingStep]:
        """Retrieve a specific step from the pipeline by name.

        Args:
            name: Name of the step to retrieve.

        Returns:
            The step associated with the given name, or None if not found.
        """
        for step_name, step in self.steps:
            if step_name == name:
                return step
        return None

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        steps_repr = "\n  ".join([f"{name}: {type(s).__name__}" for name, s in self.steps])
        return f"PreprocessingPipeline(\n  {steps_repr}\n)"
