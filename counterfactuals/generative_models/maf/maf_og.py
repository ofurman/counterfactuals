"""Implementations of autoregressive flows."""

import torch.nn.functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation


# class Dequantization(Transform):
#     def __init__(self, alpha=1e-5, quants=256):
#         """
#         Inputs:
#             alpha - small constant that is used to scale the original input.
#                     Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
#             quants - Number of possible discrete values (usually 256 for 8-bit image)
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.quants = quants

#     def forward(self, inputs, context=None):
#         ldj = z.new_zeros(z.size(0))

#         # Dequantization step
#         z = z.to(torch.float32)
#         z = z + torch.rand_like(z).detach()
#         z = z / self.quants
#         ldj -= np.log(self.quants) * np.prod(z.shape[1:])

#         # Sigmoid (with reverse=True)
#         ldj += (-z - 2 * F.softplus(-z)).sum(dim=[1, 2, 3])
#         ldj -= np.log(1 - self.alpha) * np.prod(z.shape[1:])
#         return z, ldj

#     def inverse(self, inputs, context=None):
#         ldj = z.new_zeros(z.size(0))
#         # Sigmoid (with reverse=False)
#         # Scale z to avoid boundaries 0 and 1
#         z = z * (1 - self.alpha) + 0.5 * self.alpha
#         ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
#         ldj += (-torch.log(z) - torch.log(1 - z)).sum(dim=[1, 2, 3])

#         # Inverse of the dequantization step
#         ldj += np.log(self.quants) * np.prod(z.shape[1:])
#         return z, ldj


class MaskedAutoregressiveFlow(Flow):
    """An autoregressive flow that uses affine transforms with masking.

    Reference:
    > G. Papamakarios et al., Masked Autoregressive Flow for Density Estimation,
    > Advances in Neural Information Processing Systems, 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_layers=5,
        num_blocks_per_layer=2,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):
        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []  # [Dequantization()]
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )
