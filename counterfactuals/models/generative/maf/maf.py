import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from nflows.flows import MaskedAutoregressiveFlow as _MaskedAutoregressiveFlow
from tqdm import tqdm

from counterfactuals.models.generative_mixin import GenerativePytorchMixin
from counterfactuals.models.pytorch_base import PytorchBase


class MaskedAutoregressiveFlow(PytorchBase, GenerativePytorchMixin):
    def __init__(
        self,
        num_inputs: int,
        num_targets: int,
        features: int,
        hidden_features: int,
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
        device="cpu",
    ):
        super(MaskedAutoregressiveFlow, self).__init__(num_inputs, num_targets)
        self.features = features
        self.hidden_features = hidden_features
        self.device = device
        self.context_features = context_features
        self.num_layers = num_layers
        self.num_blocks_per_layer = num_blocks_per_layer
        self.use_residual_blocks = use_residual_blocks
        self.use_random_masks = use_random_masks
        self.use_random_permutations = use_random_permutations
        self.activation = activation
        self.dropout_probability = dropout_probability
        self.batch_norm_within_layers = batch_norm_within_layers
        self.batch_norm_between_layers = batch_norm_between_layers
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
            context = context.view(-1, self.context_features)
        return self.model.log_prob(inputs=x, context=context)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
        dequantizer=None,
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
                if dequantizer:
                    inputs = dequantizer.transform(inputs.numpy())
                    inputs = torch.from_numpy(inputs)

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
        Returns a torch tensor stacked across batches.
        """
        self.eval()
        log_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.type(torch.float32)
                outputs = self(inputs, labels)
                log_probs.append(outputs)
        results = torch.hstack(log_probs)

        assert len(dataloader.dataset) == len(results)
        return results

    def sample_and_log_proba(self, n_samples: int, context: np.ndarray = None):
        """Sample from the model and return (samples, log_probs) as numpy arrays."""
        # Accept numpy arrays for context and convert to torch.tensor
        if isinstance(context, np.ndarray):
            context_tensor = torch.from_numpy(context).float()
        else:
            context_tensor = context

        if context_tensor is not None:
            context_tensor = context_tensor.view(-1, self.context_features)

        self.eval()
        with torch.no_grad():
            samples, log_probs = self.model.sample_and_log_prob(num_samples=n_samples, context=context_tensor)
            return samples.cpu().numpy(), log_probs.cpu().numpy()

    def predict_log_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict log probabilities for input data (numpy array) and return numpy array."""
        if isinstance(X_test, np.ndarray):
            X_test_tensor = torch.from_numpy(X_test).float()
        else:
            X_test_tensor = X_test

        self.eval()
        with torch.no_grad():
            log_probs = self.model.log_prob(X_test_tensor)
            return log_probs.cpu().numpy()

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

    # Compatibility with PytorchBase abstract interface -------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return point predictions for input X. For a generative flow this is not a class label
        prediction; we expose the model's log-probabilities as a sensible default.

        Accepts numpy arrays or torch tensors and returns a numpy array.
        """
        # delegate to predict_log_proba which already supports numpy inputs
        return self.predict_log_proba(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        For consistency with the classifier interface, expose probabilities derived from
        log-probabilities. Here we return normalized probabilities via softmax across a
        singleton axis so callers expecting a (N, C) shape get a 2D array. For pure density
        models this is somewhat synthetic but keeps API compatibility.
        """
        logp = self.predict_log_proba(X_test)
        # Ensure we have a 2D array (N, ) -> (N, 1)
        import numpy as _np

        logp = _np.asarray(logp)
        if logp.ndim == 1:
            # create a single-column probability by exponentiating and normalizing to 1
            probs = _np.exp(logp - _np.max(logp))
            probs = probs / probs.sum()
            # return shape (N,1)
            return probs.reshape(-1, 1)
        else:
            # if logp already multi-dimensional, softmax along last axis
            exp = _np.exp(logp - _np.max(logp, axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
