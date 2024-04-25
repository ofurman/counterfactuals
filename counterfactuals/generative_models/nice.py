import neptune
import torch
import torch.optim as optim
import torch.nn.functional as F
from counterfactuals.generative_models import BaseGenModel
from nflows.flows import SimpleRealNVP as _SimpleRealNVP

from tqdm import tqdm


class NICE(BaseGenModel):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_layers=5,
        num_blocks_per_layer=2,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        neptune_run=None,
        device="cpu",
    ):
        super(NICE, self).__init__()
        self.device = device
        self.neptune_run = neptune_run
        self.model = _SimpleRealNVP(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_layers=num_layers,
            num_blocks_per_layer=num_blocks_per_layer,
            activation=activation,
            use_volume_preserving=True,
            dropout_probability=dropout_probability,
            batch_norm_within_layers=batch_norm_within_layers,
            batch_norm_between_layers=batch_norm_between_layers,
        )

    def forward(self, x, context=None):
        if context is not None:
            context = context.view(-1, 1)
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

    def predict_log_prob(self, dataloader):
        self.eval()
        log_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.type(torch.float32)
                outputs = self(inputs, labels)
                log_probs.append(outputs)

        return torch.hstack(log_probs)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
