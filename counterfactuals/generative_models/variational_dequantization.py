import torch
import numpy as np
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

from counterfactuals.generative_models.dequantization import Dequantization


class VariationalDequantization(Dequantization):
    def __init__(
        self,
        category_counts: list[int],
        category_indices: list[int],
        features: int,
        hidden_features: int,
        context_features: int = None,
        num_blocks_per_layer: int = 2,
        use_residual_blocks: bool = True,
        use_random_masks: bool = False,
        activation: callable = torch.nn.functional.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        alpha: float = 1e-5,
    ):
        """
        Inputs:
            var_flows - A list of flow transformations to use for modeling q(u|x)
            alpha - Small constant, see Dequantization for details
        """
        super().__init__(alpha=alpha)
        self.category_counts = category_counts
        self.category_indices = category_indices
        self.flow = MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=features,
            num_blocks=num_blocks_per_layer,
            use_residual_blocks=use_residual_blocks,
            random_mask=use_random_masks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm_within_layers,
        )

    def dequant(self, z, ldj):
        z = z.to(torch.float32)
        z_scaled = []
        for i, count in enumerate(self.category_counts):
            z_feature = z[:, i].unsqueeze(1)
            z_feature = (z_feature / count) * 2 - 1
            z_scaled.append(z_feature)

        z_scaled = torch.cat(z_scaled, dim=1)
        deq_noise = torch.rand_like(z_scaled).detach()
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)

        deq_noise, flow_ldj = self.flow(deq_noise, context=z_scaled)
        ldj -= flow_ldj

        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        z_final = []
        for i, count in enumerate(self.category_counts):
            deq_noise_feature = deq_noise[:, i].unsqueeze(1)
            z_final_feature = (z[:, i].unsqueeze(1) + deq_noise_feature) / (count + 1)
            z_final.append(z_final_feature)
        z_final = torch.cat(z_final, dim=1)

        ldj -= sum(np.log(count + 1) for count in self.category_counts)
        return z_final, ldj
