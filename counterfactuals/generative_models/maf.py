import neptune
import torch
import torch.optim as optim
import torch.nn.functional as F
from counterfactuals.generative_models import BaseGenModel


from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from nflows.utils import torchutils
import nflows.utils.typechecks as check

from tqdm import tqdm


class StandardNormalWithTemp(StandardNormal):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__(shape)

    def _sample(self, num_samples, context, temp=1.0):
        if context is None:
            return (
                torch.randn(num_samples, *self._shape, device=self._log_z.device) * temp
            )
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = (
                torch.randn(
                    context_size * num_samples, *self._shape, device=context.device
                )
                * temp
            )
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def sample(self, num_samples, context=None, batch_size=None, temp=1.0):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context, temp=temp)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context, temp=temp))
            return torch.cat(samples, dim=0)

    def sample_and_log_prob(self, num_samples, context=None, temp=1.0):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, ...] if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context, temp=temp)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = torchutils.merge_leading_dims(samples, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob


class MaskedAutoregressiveFlow(Flow, BaseGenModel):
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
        self.device = device
        self.neptune_run = neptune_run
        self.context_features = context_features
        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
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
            distribution=StandardNormalWithTemp([features]),
        )

    def forward(self, x, context=None):
        if context is not None:
            context = context.view(-1, self.context_features)
        return self.log_prob(inputs=x, context=context)

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
        patience_counter = 0
        min_test_loss = float("inf")

        for epoch in (pbar := tqdm(range(num_epochs))):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.type(torch.float32)
                optimizer.zero_grad()
                log_likelihood = self(inputs, labels)
                loss = -log_likelihood.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    labels = labels.type(torch.float32)
                    log_likelihood = self(inputs, labels)
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

    def sample_and_log_prob(self, num_samples, context=None, temp=1.0):
        if context is not None:
            context = context.view(-1, self.context_features)

        embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=embedded_context, temp=temp
        )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    # Deprecated due tu multiclass support, use self.forward instead
    # def predict_log_probs(self, X: Union[np.ndarray, torch.Tensor]):
    #     """
    #     Predict log probabilities of the input dataset for both context equal 0 and 1.
    #     Results format is of the shape: [2, N]. N is number of samples, i.e., X.shape[0].
    #     """
    #     self.eval()
    #     if isinstance(X, np.ndarray):
    #         X = torch.from_numpy(X)
    #     with torch.no_grad():
    #         y_zero = torch.zeros((X.shape[0], 1), dtype=X.dtype).to(self.device)
    #         y_one = torch.ones((X.shape[0], 1), dtype=X.dtype).to(self.device)
    #         log_p_zero = self(X, y_zero)
    #         log_p_one = self(X, y_one)
    #     result = torch.vstack([log_p_zero, log_p_one])

    #     assert result.T.shape[0] == X.shape[0], f"Shape of results don't match. " \
    #                                             f"Shape of result: {result.shape}, shape of input: {X.shape}"
    #     return result

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
