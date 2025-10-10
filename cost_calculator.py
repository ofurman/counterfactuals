"""
Counterfactual Cost Calculator

This module implements the cost calculation for counterfactual explanations
based on the GLOBE-CE and AReS implementations. The cost calculation follows
these principles:

For Continuous Features:
- Continuous features are binned into 10 equal intervals post-training
- The cost of moving between adjacent bins is set to 1
- Uses ℓ1 distance (unit costs per decile of continuous features)

For Categorical Features:
- The cost of moving from one categorical feature value to another is set to 1 (unit costs)

Author: AI Assistant
Based on: GLOBE-CE and AReS implementations
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd


class CounterfactualCostCalculator:
    """
    A class to calculate the cost of changing from original instances to counterfactual instances.

    This implementation is based on the cost calculation logic found in GLOBE-CE and AReS methods.
    """

    def __init__(
        self,
        categorical_features: List[str] = None,
        ordinal_features: List[str] = None,
        n_bins: int = 10,
    ):
        """
        Initialize the cost calculator.

        Args:
            categorical_features: List of categorical feature names
            ordinal_features: List of ordinal feature names (subset of categorical)
            n_bins: Number of bins for continuous features (default: 10)
        """
        self.categorical_features = categorical_features or []
        self.ordinal_features = ordinal_features or []
        self.n_bins = n_bins

        # These will be computed during setup
        self.feature_costs_vector = None
        self.non_ordinal_categories_idx = None
        self.ordinal_categories_idx = None
        self.bin_widths = {}
        self.features_tree = {}
        self.feature_names = None
        self.any_non_ordinal = False
        self.any_ordinal = False

    def setup_from_data(
        self,
        data: pd.DataFrame,
        categorical_features: List[str] = None,
        ordinal_features: List[str] = None,
    ) -> None:
        """
        Setup the cost calculator from training data.

        Args:
            data: Training data DataFrame (features only, no target)
            categorical_features: List of categorical feature names
            ordinal_features: List of ordinal feature names
        """
        if categorical_features is not None:
            self.categorical_features = categorical_features
        if ordinal_features is not None:
            self.ordinal_features = ordinal_features

        # Process the data to understand feature structure
        self._process_features(data)
        self._setup_cost_vectors()

    def _process_features(self, data: pd.DataFrame) -> None:
        """Process features to understand their structure and create binning for continuous features."""
        self.features_tree = {}
        self.bin_widths = {}
        processed_features = []

        for col in data.columns:
            self.features_tree[col] = []

            if col in self.categorical_features:
                # Categorical feature - store unique values
                unique_vals = data[col].unique()
                for val in unique_vals:
                    feature_value = f"{col} = {val}"
                    processed_features.append(feature_value)
                    self.features_tree[col].append(feature_value)
            else:
                # Continuous feature - create bins
                _, bins = pd.cut(data[col], bins=self.n_bins, retbins=True)
                bin_width = bins[1] - bins[0]  # Assuming equal width bins
                self.bin_widths[col] = bin_width

                # Create bin categories
                binned_data = pd.cut(data[col], bins=self.n_bins)
                for interval in binned_data.cat.categories:
                    feature_value = f"{col} = {interval}"
                    processed_features.append(feature_value)
                    self.features_tree[col].append(feature_value)

        self.feature_names = processed_features

    def _setup_cost_vectors(self) -> None:
        """Setup the cost vectors based on feature types."""
        n_features = len(self.feature_names)
        self.feature_costs_vector = np.zeros(n_features)
        self.non_ordinal_categories_idx = np.ones(n_features, dtype=bool)

        i = 0
        for feature in self.features_tree:
            if feature not in self.categorical_features:
                # Continuous feature
                if feature in self.bin_widths:
                    self.feature_costs_vector[i] = 1 / self.bin_widths[feature]
                else:
                    self.feature_costs_vector[i] = 1
                self.non_ordinal_categories_idx[i] = True
                i += 1
            else:
                # Categorical feature
                n = len(self.features_tree[feature])
                if feature in self.ordinal_features:
                    # Ordinal feature - cost is the distance between levels
                    self.feature_costs_vector[i : i + n] = np.arange(n)
                    self.non_ordinal_categories_idx[i : i + n] = False
                else:
                    # Non-ordinal categorical - cost 1 for any change (2 changes of 0.5)
                    self.feature_costs_vector[i : i + n] = 0.5
                    self.non_ordinal_categories_idx[i : i + n] = True
                i += n

        self.ordinal_categories_idx = ~self.non_ordinal_categories_idx
        self.any_non_ordinal = self.non_ordinal_categories_idx.any()
        self.any_ordinal = self.ordinal_categories_idx.any()

    def compute_costs(
        self, original_instances: np.ndarray, counterfactual_instances: np.ndarray
    ) -> np.ndarray:
        """
        Compute the costs of changing from original to counterfactual instances.

        Args:
            original_instances: Original instances (n_samples, n_features)
            counterfactual_instances: Counterfactual instances (n_samples, n_features)

        Returns:
            Array of costs for each instance

        Note:
            Both arrays should be in the same one-hot encoded format used during training.
        """
        if self.feature_costs_vector is None:
            raise ValueError("Cost calculator not setup. Call setup_from_data() first.")

        # Ensure inputs are numpy arrays
        original_instances = np.asarray(original_instances)
        counterfactual_instances = np.asarray(counterfactual_instances)

        if original_instances.shape != counterfactual_instances.shape:
            raise ValueError(
                "Original and counterfactual instances must have the same shape"
            )

        # Calculate the difference
        x_diff = counterfactual_instances - original_instances
        ret = np.zeros(original_instances.shape[0])

        if self.any_non_ordinal:
            # For non-ordinal features (continuous and non-ordinal categorical)
            # Use L1 norm: sum(abs(differences)) weighted by costs
            ret += np.linalg.norm(
                x_diff[:, self.non_ordinal_categories_idx]
                * self.feature_costs_vector[self.non_ordinal_categories_idx],
                axis=1,
                ord=1,
            )

        if self.any_ordinal:
            # For ordinal features: abs(sum(weighted_differences))
            # This accounts for the ordering in ordinal features
            ret += np.abs(
                (
                    x_diff[:, self.ordinal_categories_idx]
                    * self.feature_costs_vector[self.ordinal_categories_idx]
                ).sum(1)
            )

        return ret

    def compute_feature_wise_costs(
        self, original_instances: np.ndarray, counterfactual_instances: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature-wise costs for better interpretability.

        Args:
            original_instances: Original instances (n_samples, n_features)
            counterfactual_instances: Counterfactual instances (n_samples, n_features)

        Returns:
            Dictionary mapping feature names to their costs per instance
        """
        if self.feature_costs_vector is None:
            raise ValueError("Cost calculator not setup. Call setup_from_data() first.")

        original_instances = np.asarray(original_instances)
        counterfactual_instances = np.asarray(counterfactual_instances)
        x_diff = counterfactual_instances - original_instances

        feature_costs = {}
        i = 0

        for feature in self.features_tree:
            if feature not in self.categorical_features:
                # Continuous feature
                cost = np.abs(x_diff[:, i] * self.feature_costs_vector[i])
                feature_costs[feature] = cost
                i += 1
            else:
                # Categorical feature
                n = len(self.features_tree[feature])
                feature_diff = x_diff[:, i : i + n]
                feature_cost_vec = self.feature_costs_vector[i : i + n]

                if feature in self.ordinal_features:
                    # Ordinal: absolute sum of weighted differences
                    cost = np.abs((feature_diff * feature_cost_vec).sum(1))
                else:
                    # Non-ordinal: L1 norm of weighted differences
                    cost = np.linalg.norm(
                        feature_diff * feature_cost_vec, axis=1, ord=1
                    )

                feature_costs[feature] = cost
                i += n

        return feature_costs


def calculate_counterfactual_costs(
    original_instances: Union[pd.DataFrame, np.ndarray],
    counterfactual_instances: Union[pd.DataFrame, np.ndarray],
    training_data: pd.DataFrame = None,
    categorical_features: List[str] = None,
    ordinal_features: List[str] = None,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Convenience function to calculate counterfactual costs.

    Args:
        original_instances: Original instances
        counterfactual_instances: Counterfactual instances
        training_data: Training data to setup cost calculation (required if instances are not one-hot encoded)
        categorical_features: List of categorical feature names
        ordinal_features: List of ordinal feature names
        n_bins: Number of bins for continuous features

    Returns:
        Array of costs for each instance

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Sample training data
        >>> training_data = pd.DataFrame({
        ...     'age': [25, 30, 35, 40, 45],
        ...     'income': [30000, 40000, 50000, 60000, 70000],
        ...     'education': ['high_school', 'bachelor', 'master', 'bachelor', 'phd']
        ... })
        >>>
        >>> # Define categorical features
        >>> categorical_features = ['education']
        >>>
        >>> # Original and counterfactual instances (should be one-hot encoded)
        >>> # This is a simplified example - in practice, you'd have proper one-hot encoding
        >>> original = np.array([[0.3, 0.4, 1, 0, 0, 0, 0]])  # One instance
        >>> counterfactual = np.array([[0.5, 0.6, 0, 1, 0, 0, 0]])  # Changed education
        >>>
        >>> # Calculate costs
        >>> costs = calculate_counterfactual_costs(
        ...     original, counterfactual, training_data, categorical_features
        ... )
        >>> print(f"Cost: {costs[0]:.2f}")
    """
    calculator = CounterfactualCostCalculator(
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        n_bins=n_bins,
    )

    if training_data is not None:
        calculator.setup_from_data(
            training_data, categorical_features, ordinal_features
        )
    else:
        raise ValueError("training_data is required to setup the cost calculator")

    return calculator.compute_costs(original_instances, counterfactual_instances)


if __name__ == "__main__":
    # Example usage
    print("Counterfactual Cost Calculator")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    training_data = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 100),
            "income": np.random.normal(50000, 15000, 100),
            "education": np.random.choice(
                ["high_school", "bachelor", "master", "phd"], 100
            ),
            "experience": np.random.normal(10, 5, 100),
        }
    )

    print("Sample training data shape:", training_data.shape)
    print("Categorical features: ['education']")
    print("Continuous features: ['age', 'income', 'experience']")

    # Initialize calculator
    calculator = CounterfactualCostCalculator(
        categorical_features=["education"], n_bins=10
    )

    # Setup from training data
    calculator.setup_from_data(training_data)

    print("\nSetup complete!")
    print(f"Number of processed features: {len(calculator.feature_names)}")
    print(f"Any non-ordinal features: {calculator.any_non_ordinal}")
    print(f"Any ordinal features: {calculator.any_ordinal}")

    # Note: In practice, you would have properly one-hot encoded data
    # This is just a demonstration of the function interface
    print("\nCost calculator is ready to use!")
    print("Use calculator.compute_costs(original, counterfactual) to calculate costs.")
