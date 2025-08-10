from typing import Optional, Tuple, Union

import neptune
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from counterfactuals.generative_models import BaseGenModel
from nflows.flows import MaskedAutoregressiveFlow as _MaskedAutoregressiveFlow


class MaskedAutoregressiveFlow(BaseGenModel):
    """Masked Autoregressive Flow (MAF) generative model.
    
    A normalizing flow that learns complex probability distributions by transforming
    a simple base distribution through invertible autoregressive transformations.
    """
    
    def __init__(
        self,
        features: int,
        hidden_features: int,
        context_features: Optional[int] = None,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        use_residual_blocks: bool = True,
        use_random_masks: bool = False,
        use_random_permutations: bool = False,
        activation: callable = F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        batch_norm_between_layers: bool = False,
        neptune_run: Optional["neptune.Run"] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the Masked Autoregressive Flow (MAF) model.

        Args:
            features (int): Number of input features.
            hidden_features (int): Number of hidden units in each transformation network.
            context_features (Optional[int], optional): Number of context features for conditional modeling. Defaults to None.
            num_layers (int, optional): Number of autoregressive layers. Defaults to 5.
            num_blocks_per_layer (int, optional): Number of transformation blocks per layer. Defaults to 2.
            use_residual_blocks (bool, optional): Whether to use residual connections within blocks. Defaults to True.
            use_random_masks (bool, optional): Whether to use random autoregressive masks. Defaults to False.
            use_random_permutations (bool, optional): Whether to randomly permute features between layers. Defaults to False.
            activation (callable, optional): Activation function for transformation networks. Defaults to torch.nn.functional.relu.
            dropout_probability (float, optional): Dropout rate for regularization. Defaults to 0.0.
            batch_norm_within_layers (bool, optional): Whether to apply batch normalization within layers. Defaults to False.
            batch_norm_between_layers (bool, optional): Whether to apply batch normalization between layers. Defaults to False.
            neptune_run (Optional[neptune.Run], optional): Neptune run object for experiment tracking. Defaults to None.
            device (str, optional): Device to run computations on (e.g., "cpu" or "cuda"). Defaults to "cpu".
        """
        
        super(MaskedAutoregressiveFlow, self).__init__()
        self.device = device
        self.neptune_run = neptune_run
        self.context_features = context_features
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

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log probabilities for input data.
        
        Args:
            x: Input tensor of shape (batch_size, features).
            context: Optional context tensor of shape (batch_size, context_features).
            
        Returns:
            Log probabilities for each input sample.
        """
        if context is not None:
            context = context.view(-1, self.context_features)
        return self.model.log_prob(inputs=x, context=context)

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
        neptune_run: Optional[neptune.Run] = None,
        dequantizer=None,
    ) -> None:
        """Train the MAF model on provided data.
        
        Args:
            train_loader: DataLoader for training data.
            test_loader: DataLoader for validation data.
            num_epochs: Maximum number of training epochs.
            learning_rate: Learning rate for Adam optimizer.
            patience: Early stopping patience (epochs without improvement).
            eps: Minimum improvement threshold for early stopping.
            checkpoint_path: Path to save best model checkpoint.
            neptune_run: Neptune run for logging metrics.
            dequantizer: Optional dequantizer for data preprocessing.
        """
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
                    numerical_features = inputs[:, dequantizer.dropped_numerical]
                    inputs = dequantizer.transform(inputs.numpy())
                    inputs = torch.from_numpy(inputs)
                    inputs = torch.cat([numerical_features, inputs], dim=1)

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

    def predict_log_prob(self, dataloader: DataLoader) -> torch.Tensor:
        """Compute log probabilities for dataset using context from dataloader.
        
        Args:
            dataloader: DataLoader containing inputs and context labels.
            
        Returns:
            Tensor of log probabilities for all samples in dataset.
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

    def sample_and_log_prob(
        self, 
        num_samples: int, 
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples and their log probabilities.
        
        Args:
            num_samples: Number of samples to generate.
            context: Optional context tensor for conditional sampling.
            
        Returns:
            Tuple of (samples, log_probabilities).
        """
        if context is not None:
            context = context.view(-1, self.context_features)
        return self.model.sample_and_log_prob(num_samples=num_samples, context=context)

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

    def save(self, path: str) -> None:
        """Save model state to file.
        
        Args:
            path: File path to save model state.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model state from file.
        
        Args:
            path: File path to load model state from.
        """
        self.load_state_dict(torch.load(path))

    def _unpack_batch(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unpack batch data and move to device.
        
        Args:
            batch: Either a tensor or tuple of (inputs, labels).
            
        Returns:
            Tuple of (inputs, labels) moved to device.
        """
        if isinstance(batch, tuple):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
        else:
            inputs, labels = batch[0], None
            inputs = inputs.to(self.device)
        return inputs, labels
