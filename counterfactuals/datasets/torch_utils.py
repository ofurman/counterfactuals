import torch
import torch.nn as nn
from typing import List, Optional


ALPHA = 1e-6


class TorchCategoricalTransformer(nn.Module):
    def __init__(
        self, dividers: List[int], feature_indices: Optional[List[int]] = None
    ):
        """
        Initialize the transformer with dividers for each feature.

        Parameters:
        -----------
        dividers : List[int]
            List of category counts for each feature
        feature_indices : Optional[List[int]]
            Indices of categorical features in the original data.
            If None, assumes all features are categorical.
        """
        super().__init__()
        self.dividers = dividers
        self.feature_indices = feature_indices
        self.register_buffer("alpha", torch.tensor(ALPHA))

    def _dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to data to dequantize them.
        Ensures the output stays in the valid range [0, 1].

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor to dequantize

        Returns:
        --------
        torch.Tensor
            Dequantized tensor
        """
        # Create a new tensor while preserving gradients
        result = x.detach().clone()
        result.requires_grad_(x.requires_grad)

        # If feature_indices is provided, only transform those features
        feature_range = range(len(self.dividers))
        indices_to_transform = (
            self.feature_indices if self.feature_indices is not None else feature_range
        )

        for i, feature_idx in enumerate(indices_to_transform):
            # Generate noise using PyTorch's random generator
            noise = self._sigmoid(torch.randn(x.size(0), device=x.device))
            # Add noise and normalize
            data_with_noise = x[:, feature_idx] + noise
            result[:, feature_idx] = data_with_noise / self.dividers[i]

        # The transformed features are differentiable, and we preserve the original values
        # for non-transformed features to maintain gradient flow
        if self.feature_indices is not None:
            non_transformed_indices = [
                i for i in range(x.shape[1]) if i not in self.feature_indices
            ]
            if non_transformed_indices:
                result[:, non_transformed_indices] = x[:, non_transformed_indices]

        return result

    def _logit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms values with logit to be unconstrained.
        Applied only to the categorical features.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Logit-transformed tensor
        """
        # Start with a differentiable copy of the input
        result = torch.zeros_like(x)

        # If feature_indices is provided, only transform those features
        indices_to_transform = (
            self.feature_indices
            if self.feature_indices is not None
            else range(x.shape[1])
        )

        for feature_idx in indices_to_transform:
            # Apply logit transform only to the categorical features
            feature_data = x[:, feature_idx]
            transformed = self.alpha + (1 - 2 * self.alpha) * feature_data
            result[:, feature_idx] = torch.log(transformed / (1.0 - transformed))

        # For non-transformed features, use original values to maintain gradient flow
        if self.feature_indices is not None:
            non_transformed_indices = [
                i for i in range(x.shape[1]) if i not in self.feature_indices
            ]
            if non_transformed_indices:
                result[:, non_transformed_indices] = x[:, non_transformed_indices]

        return result

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid function implemented in PyTorch.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Tensor after sigmoid transformation
        """
        return torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input tensor by dequantizing and applying logit transform
        only to the categorical features.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Transformed tensor
        """
        x_transformed = self._dequantize(x)
        x_transformed = self._logit_transform(x_transformed)
        return x_transformed

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform the logit transformation.
        Applied only to the categorical features.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Inverse transformed tensor with categorical values
        """
        # Create result tensor while preserving differentiability
        result = torch.zeros_like(x)

        # If feature_indices is provided, only transform those features
        indices_to_transform = (
            self.feature_indices
            if self.feature_indices is not None
            else range(x.shape[1])
        )

        for i, feature_idx in enumerate(indices_to_transform):
            # Apply inverse transform only to the categorical features
            feature_data = x[:, feature_idx]
            # Sigmoid and scale
            feature_data = (torch.sigmoid(feature_data) - 1e-6) / (1 - 2e-6)

            # Digitize operation
            bins = torch.linspace(0, 1, self.dividers[i] + 1, device=x.device)
            bin_indices = torch.sum(feature_data.unsqueeze(1) >= bins, dim=1) - 1

            # Store the result - note that this is not differentiable due to binning
            result[:, feature_idx] = bin_indices.float()

        # For non-transformed features, preserve original values
        if self.feature_indices is not None:
            non_transformed_indices = [
                i for i in range(x.shape[1]) if i not in self.feature_indices
            ]
            if non_transformed_indices:
                result[:, non_transformed_indices] = x[:, non_transformed_indices]

        return result

    @staticmethod
    def from_numpy_transformer(
        transformer, feature_indices=None
    ) -> "TorchCategoricalTransformer":
        """
        Create a TorchCategoricalTransformer from a fitted CustomCategoricalTransformer.

        Parameters:
        -----------
        transformer : CustomCategoricalTransformer
            Fitted sklearn transformer
        feature_indices : Optional[List[int]]
            Indices of categorical features in the original data

        Returns:
        --------
        TorchCategoricalTransformer
            PyTorch version of the transformer
        """
        return TorchCategoricalTransformer(
            dividers=transformer.dividers, feature_indices=feature_indices
        )


def dequantize_torch(
    x: torch.Tensor, transformer: TorchCategoricalTransformer
) -> torch.Tensor:
    """
    Apply dequantization to a PyTorch tensor using the transformer.
    Only the categorical features specified in the transformer will be transformed.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor
    transformer : TorchCategoricalTransformer
        Transformer to use for dequantization

    Returns:
    --------
    torch.Tensor
        Dequantized tensor with only categorical features transformed
    """
    return transformer(x)


def inverse_dequantize_torch(
    x: torch.Tensor, transformer: TorchCategoricalTransformer
) -> torch.Tensor:
    """
    Apply inverse dequantization to a PyTorch tensor using the transformer.
    Only the categorical features specified in the transformer will be transformed.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor
    transformer : TorchCategoricalTransformer
        Transformer to use for inverse dequantization

    Returns:
    --------
    torch.Tensor
        Inverse dequantized tensor with only categorical features transformed
    """
    return transformer.inverse_transform(x)


class DifferentiableSelectiveDequantizer(nn.Module):
    """
    A differentiable utility class that helps selectively dequantize categorical features
    while preserving continuous features.
    """

    def __init__(
        self,
        categorical_feature_lists: List[List[int]],
        dividers_list: Optional[List[List[int]]] = None,
    ):
        """
        Initialize the selective dequantizer.

        Parameters:
        -----------
        categorical_feature_lists : List[List[int]]
            Lists of categorical feature indices for each group
        dividers_list : Optional[List[List[int]]]
            Lists of dividers for each categorical feature group.
            If None, dividers will need to be fitted.
        """
        super().__init__()
        self.categorical_feature_lists = categorical_feature_lists
        self.dividers_list = dividers_list
        self.transformers = nn.ModuleList()

        # Create transformers for each feature group if dividers are provided
        if dividers_list is not None:
            for feature_list, dividers in zip(categorical_feature_lists, dividers_list):
                self.transformers.append(
                    TorchCategoricalTransformer(
                        dividers=dividers, feature_indices=feature_list
                    )
                )

    def fit(self, x: torch.Tensor):
        """
        Fit the transformer by determining the dividers for each categorical feature.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor to fit on

        Returns:
        --------
        self
        """
        self.transformers = nn.ModuleList()
        self.dividers_list = []

        for feature_list in self.categorical_feature_lists:
            # Calculate dividers for each feature in the group
            dividers = []
            for feature_idx in feature_list:
                # Get max value to determine number of categories
                max_val = int(x[:, feature_idx].max().item()) + 1
                dividers.append(max_val)

            self.dividers_list.append(dividers)
            # Create a transformer for this feature group
            self.transformers.append(
                TorchCategoricalTransformer(
                    dividers=dividers, feature_indices=feature_list
                )
            )

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input tensor by applying all transformers.
        Preserves gradients.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Transformed tensor
        """
        result = x

        for transformer in self.transformers:
            # Apply each transformer (they only modify their specific features)
            result = transformer(result)

        return result

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform the input tensor by applying all inverse transformers.
        Note: This operation is generally not differentiable due to the binning.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor

        Returns:
        --------
        torch.Tensor
            Inverse transformed tensor
        """
        result = x

        for transformer in self.transformers:
            # Apply each inverse transformer
            result = transformer.inverse_transform(result)

        return result
