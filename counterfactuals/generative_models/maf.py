"""Implementations of autoregressive flows."""

import neptune
import torch
import torch.optim as optim
import torch.nn.functional as F
from counterfactuals.generative_models import BaseGenModel

from tqdm import tqdm

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation

from counterfactuals.generative_models.dequantization import Dequantization


class MaskedAutoregressiveFlow(Flow, BaseGenModel):
    """An autoregressive flow that uses affine transforms with masking.

    Reference:
    > G. Papamakarios et al., Masked Autoregressive Flow for Density Estimation,
    > Advances in Neural Information Processing Systems, 2017.
    """

    def __init__(
        self,
        features,
        hidden_features,
        category_counts,
        category_indices,
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
        neptune_run=None,
        device="cpu",
    ):
        self.device = device
        self.neptune_run = neptune_run

        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        # layers.append(
        #     VariationalDequantization(
        #         category_counts=category_counts,
        #         category_indices=category_indices,
        #         features=features,
        #         hidden_features=hidden_features,
        #         context_features=features,
        #         num_blocks_per_layer=num_blocks_per_layer,
        #         use_residual_blocks=use_residual_blocks,
        #         use_random_masks=use_random_masks,
        #         activation=activation,
        #         dropout_probability=dropout_probability,
        #         batch_norm_within_layers=batch_norm_within_layers,
        #     )
        # )
        layers.append(
            Dequantization(
                category_counts=category_counts,
                category_indices=category_indices,
            )
        )
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

    def forward(self, inputs, context=None):
        return self._transform(inputs, context=context)

    def log_prob_(self, x, context=None, without_dequantization=False):
        if context is not None:
            context = context.view(-1, 1)
        if without_dequantization:
            self._transform_all = self._transform._transforms
            self._transform._transforms = self._transform._transforms[1:]
            log_prob = self.log_prob(inputs=x, context=context)
            self._transform._transforms = self._transform_all
        else:
            log_prob = self.log_prob(inputs=x, context=context)
        return log_prob

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
        neptune_run: neptune.Run = None,
    ):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in (pbar := tqdm(range(num_epochs))):
            self.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_likelihood = self.log_prob_(inputs, labels)
                loss = -log_likelihood.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()
            test_loss = 0.0
            min_test_loss = float("inf")
            with torch.no_grad():
                for inputs, labels in test_loader:
                    log_likelihood = self.log_prob_(inputs, labels)
                    loss = -log_likelihood.mean().item()
                    test_loss += loss
            test_loss /= len(test_loader)
            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, test: {test_loss:.4f}"
            )
            if neptune_run:
                neptune_run["gen_train_nll"].append(train_loss)
                neptune_run["gen_test_nll"].append(test_loss)

            if test_loss - min_test_loss < eps:
                min_test_loss = test_loss
                patience_counter = 0
                self.save(checkpoint_path)

            else:
                patience_counter += 1
            if patience_counter > patience:
                break
        self.load(checkpoint_path)

    def predict_log_prob(self, dataloader):
        self.eval()
        log_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self.log_prob_(inputs, labels)
                log_probs.append(outputs)

        return torch.hstack(log_probs)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
