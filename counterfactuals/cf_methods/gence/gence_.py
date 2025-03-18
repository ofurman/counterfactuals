import neptune
import torch
import torch.optim as optim
import torch.nn.functional as F
from counterfactuals.generative_models import BaseGenModel
from counterfactuals.generative_models.maf import (
    MaskedAutoregressiveFlow as _MaskedAutoregressiveFlow,
)
# from nflows.flows import MaskedAutoregressiveFlow as _MaskedAutoregressiveFlow

from tqdm import tqdm


class GenCE(BaseGenModel):
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
        neptune_run=None,
        device="cpu",
    ):
        super(GenCE, self).__init__()
        self.device = device
        self.neptune_run = neptune_run
        self.model = _MaskedAutoregressiveFlow(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_layers=num_layers,
            num_blocks_per_layer=num_blocks_per_layer,
            use_residual_blocks=use_residual_blocks,
            use_random_masks=use_random_masks,
            use_random_permutations=use_random_permutations,
            activation=activation,
            dropout_probability=dropout_probability,
            batch_norm_within_layers=batch_norm_within_layers,
            batch_norm_between_layers=batch_norm_between_layers,
        )

    def forward(self, x, context=None):
        if context is not None:
            context = context.view(-1, 2)
        return self.model.log_prob(inputs=x, context=context)

    # def scale_nll(self, nll, x_cf, x_orig, alpha):
    #     dist = torch.linalg.norm(x_cf - x_orig, axis=1, ord=2)
    #     return nll / (dist**2)

    def scale_nll(self, nll, x_cf, x_orig, scale_nll_fn):
        dist = torch.linalg.norm(x_cf - x_orig, axis=1, ord=2)
        return nll * scale_nll_fn(dist)

    def scale_nll_gaussian_kernel(self, nll, x_cf, x_orig, alpha=4):
        dist = torch.linalg.norm(x_cf - x_orig, axis=1, ord=2)
        return nll * torch.exp(-alpha * dist**2)

    def sample_and_log_prob(self, num_samples, context=None):
        if context is not None:
            context = context.view(-1, 2)
        return self.model.sample_and_log_prob(num_samples=num_samples, context=context)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        alpha: float = 1.0,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
        scale_nll_fn=None,
        neptune_run: neptune.Run = None,
    ):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        patience_counter = 0
        min_test_loss = float("inf")

        for epoch in (pbar := tqdm(range(num_epochs))):
            self.train()
            train_loss = 0.0
            for x_cf, x_orig in train_loader:
                x_cf, x_orig = x_cf.to(self.device), x_orig.to(self.device)
                optimizer.zero_grad()
                log_likelihood = self(x_cf, x_orig)
                log_likelihood = self.scale_nll(
                    log_likelihood, x_cf, x_orig, scale_nll_fn
                )
                loss = -log_likelihood.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()
            test_loss = 0.0
            with torch.no_grad():
                for x_cf, x_orig in test_loader:
                    log_likelihood = self(x_cf, x_orig)
                    log_likelihood = self.scale_nll(
                        log_likelihood, x_cf, x_orig, scale_nll_fn
                    )
                    loss = -log_likelihood.mean().item()
                    test_loss += loss
            test_loss /= len(test_loader)
            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, test: {test_loss:.4f}, patience: {patience_counter}"
            )
            if neptune_run:
                neptune_run["gen_train_nll"].append(train_loss)
                neptune_run["gen_test_nll"].append(test_loss)
            if test_loss < (min_test_loss + eps):
                min_test_loss = test_loss
                patience_counter = 0
                self.save(checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter > patience:
                break
        self.load(checkpoint_path)

    def predict_log_prob(self, dataloader) -> torch.Tensor:
        """
        Predict log probabilities for the given dataset using the context included in the dataset.
        """
        self.eval()
        log_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.type(torch.float32)
                outputs = self(inputs, labels)
                log_probs.append(outputs)
        results = torch.concat(log_probs)

        assert len(dataloader.dataset) == len(results)
        return results

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def _unpack_batch(self, batch):
        if isinstance(batch, tuple):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
        else:
            inputs, labels = batch[0], None
            inputs = inputs.to(self.device)
        return inputs, labels
