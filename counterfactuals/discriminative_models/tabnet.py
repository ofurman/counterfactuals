import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm

from counterfactuals.discriminative_models.base import BaseDiscModel

##############################################################################
# Supporting Layers/Functions
##############################################################################


class GhostBatchNorm(nn.Module):
    """
    BatchNorm that can handle 'virtual' small batch sizes
    by splitting a mini-batch into sub-batches (ghost batches).
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)

    def forward(self, x):
        # x: [B, C], we split along B dimension in chunks if B > virtual_batch_size
        chunks = torch.split(x, self.virtual_batch_size, dim=0)
        normed = []
        for chunk in chunks:
            normed.append(self.bn(chunk))
        return torch.cat(normed, dim=0)


class Sparsemax(nn.Module):
    """
    Sparsemax activation as used in TabNet for the attentive transformer.
    Reference: https://arxiv.org/abs/1602.02068
    """

    def forward(self, x, dim=-1):
        # 1) sort x along dim
        sorted_x, _ = torch.sort(x, descending=True, dim=dim)
        # 2) cumsum_x along dim
        cumsum_x = torch.cumsum(sorted_x, dim=dim)
        # 3) create range [1..K]
        r = torch.arange(1, x.shape[dim] + 1, device=x.device).view(
            [1] * (len(x.shape) - 1) + [-1]
        )
        if dim != -1:
            r = r.transpose(0, dim)
        # 4) find k(z) => largest k where sorted_x_k + 1/k * (1 - cumsum_x_k) > 0
        support = sorted_x + (1.0 / r) * (1 - cumsum_x) > 0
        k = torch.sum(support, dim=dim, keepdim=True).clamp(min=1)
        # 5) compute tau
        idx = k.long() - 1
        if dim == 1:
            idx = idx.expand(-1, sorted_x.shape[dim])
        tau = 1.0 / k * (1 - torch.gather(cumsum_x, dim, idx))
        tau = tau.expand_as(x)
        # 6) ReLU(x - tau)
        return torch.relu(x - tau)


def make_mlp(
    num_features, n_hidden=64, n_glu_layers=2, virtual_batch_size=128, momentum=0.02
):
    """
    Build a simple MLP block used inside FeatureTransformer blocks (GLU-like).
    Each layer uses GhostBatchNorm, linear, GLU, etc.
    """
    layers = []
    current_dim = num_features
    for _ in range(n_glu_layers):
        # We do "2 * n_hidden" because we split into (A|B) for the GLU: A * sigmoid(B)
        layers.append(nn.Linear(current_dim, 2 * n_hidden, bias=False))
        layers.append(
            GhostBatchNorm(
                2 * n_hidden, virtual_batch_size=virtual_batch_size, momentum=momentum
            )
        )
        current_dim = n_hidden  # after GLU, dimension is n_hidden
    return nn.Sequential(*layers)


class GLUBlock(nn.Module):
    """
    Applies a sequence of GLU layers:
      For each layer: x -> Linear -> BN -> (split) -> A * sigmoid(B)
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        for i in range(0, len(self.backbone), 2):
            linear = self.backbone[i]
            bn = self.backbone[i + 1]
            x = linear(x)
            x = bn(x)
            # Split into (A,B)
            a, b = x.chunk(2, dim=-1)
            x = a * torch.sigmoid(b)
        return x


##############################################################################
# TabNet Core Modules
##############################################################################


class AttentiveTransformer(nn.Module):
    """
    Computes the feature selection mask: M = Sparsemax( (fc(BN(a))) * prior ).
    a: [B, n_a]
    prior: [B, input_dim]
    returns: mask [B, input_dim]
    """

    def __init__(self, n_a, input_dim, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        # BN matches n_a
        self.bn = GhostBatchNorm(
            n_a, virtual_batch_size=virtual_batch_size, momentum=momentum
        )
        # FC goes from n_a -> input_dim
        self.fc = nn.Linear(n_a, input_dim, bias=False)
        self.spmax = Sparsemax()

    def forward(self, a, prior):
        x = self.bn(a)  # [B, n_a]
        x = self.fc(x)  # [B, input_dim]
        x = x * prior  # [B, input_dim]
        mask = self.spmax(x, dim=-1)
        return mask


class FeatureTransformer(nn.Module):
    """
    Transforms features (either shared or step-specific).
    """

    def __init__(
        self, in_dim, out_dim, n_glu_layers=2, virtual_batch_size=128, momentum=0.02
    ):
        super().__init__()
        backbone = make_mlp(
            in_dim,
            n_hidden=out_dim,
            n_glu_layers=n_glu_layers,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )
        self.glu_block = GLUBlock(backbone)

    def forward(self, x):
        return self.glu_block(x)


class TabNetBackbone(nn.Module):
    """
    Minimal TabNet backbone that keeps output dimension = (n_d + n_a) each step.

    Steps:
      1) shared_out = self.shared(x) => [B, (n_d+n_a)]
      2) for step in range(n_steps):
         d = shared_out[:, :n_d]
         a = shared_out[:, n_d:]
         mask = AttentiveTransformer(a, prior) => [B, input_dim]
         x_masked = x * mask
         prior = prior * (gamma - mask)
         out = step_feature_transformers[step](shared_out) => [B, (n_d+n_a)]
         d_out = out[:, :n_d]
         a_out = out[:, n_d:]
         collect d_out into outputs
         shared_out = out  # keep dimension = [B, (n_d+n_a)]
      3) concat all d_out => final shape = [B, n_steps * n_d].
    """

    def __init__(
        self,
        input_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_shared=2,
        n_independent=2,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        # Shared initial feature transform => [n_d + n_a]
        self.shared = FeatureTransformer(
            in_dim=input_dim,
            out_dim=n_d + n_a,
            n_glu_layers=n_shared,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

        # Step-specific transforms => each returns [n_d + n_a]
        self.step_feature_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.step_feature_transformers.append(
                FeatureTransformer(
                    in_dim=n_d + n_a,
                    out_dim=n_d + n_a,
                    n_glu_layers=n_independent,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                )
            )

        # Attentive transformers => each transforms [B, n_a] -> [B, input_dim]
        self.attentive_transformers = nn.ModuleList()
        for _ in range(n_steps):
            self.attentive_transformers.append(
                AttentiveTransformer(
                    n_a=n_a,
                    input_dim=input_dim,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum,
                )
            )

        # Initial BN
        self.initial_bn = GhostBatchNorm(
            input_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

    def forward(self, x):
        x = self.initial_bn(x)  # [B, input_dim]
        prior = torch.ones_like(x)  # [B, input_dim]

        # Shared transform => [B, n_d + n_a]
        shared_out = self.shared(x)
        outputs = []

        for step_i in range(self.n_steps):
            d = shared_out[:, : self.n_d]  # [B, n_d]
            a = shared_out[:, self.n_d :]  # [B, n_a]

            mask = self.attentive_transformers[step_i](a, prior)  # [B, input_dim]
            x_masked = x * mask
            prior = prior * (self.gamma - mask)

            # step transform => [B, (n_d + n_a)]
            out = self.step_feature_transformers[step_i](shared_out)

            d_out = out[:, : self.n_d]  # [B, n_d]
            a_out = out[:, self.n_d :]  # [B, n_a]
            outputs.append(d_out)

            # Keep dimension => no sum
            shared_out = out

        # final shape => [B, n_steps * n_d]
        return torch.cat(outputs, dim=1)


##############################################################################
# Main TabNet Class for Classification
##############################################################################


class TabNetDiscModel(BaseDiscModel):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_shared: int = 2,
        n_independent: int = 2,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        device: str = "cpu",
    ):
        """
        A TabNet-based classifier/regressor that mirrors the same method
        signatures as the MLP-based model in your library.
        """
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.target_size = target_size

        # Build the TabNet backbone
        self.backbone = TabNetBackbone(
            input_dim=input_size,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_shared=n_shared,
            n_independent=n_independent,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

        # Final linear layer => [B, target_size]
        self.final_layer = nn.Linear(n_d * n_steps, target_size)

        # If binary classification or multi-class:
        if target_size == 1:
            # Binary
            self.final_activation = nn.Sigmoid()
            self.criterion = nn.BCEWithLogitsLoss()
            # BCE expects float labels of shape [B, 1]
            self.prep_for_loss = lambda x: x.view(-1, 1).float()
        else:
            # Multi-class
            self.final_activation = nn.Softmax(dim=1)
            self.criterion = nn.CrossEntropyLoss()
            # CE expects long labels of shape [B]
            self.prep_for_loss = lambda x: x.view(-1).long()

        self.to(self.device)

    def forward(self, x):
        tabnet_out = self.backbone(x)  # [B, n_d * n_steps]
        logits = self.final_layer(tabnet_out)  # [B, target_size]
        return logits

    def fit(
        self,
        train_loader,
        test_loader=None,
        epochs=200,
        lr=0.001,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_tabnet_model.pth",
    ):
        min_test_loss = float("inf")
        patience_counter = 0

        optimizer = torch.optim.RAdam(self.parameters(), lr=lr)

        for epoch in (pbar := tqdm(range(epochs))):
            self.train()
            train_loss = 0.0
            test_loss = 0.0

            # Training loop
            for i, (examples, labels) in enumerate(train_loader):
                examples = examples.float().to(self.device)
                # If dataset's labels are one-hot, do argmax
                if labels.ndim > 1 and labels.shape[-1] > 1:
                    labels = torch.argmax(labels, dim=1)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.forward(examples)
                loss = self.criterion(logits, self.prep_for_loss(labels))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation loop
            if test_loader is not None:
                self.eval()
                with torch.no_grad():
                    for i, (examples, labels) in enumerate(test_loader):
                        examples = examples.float().to(self.device)
                        if labels.ndim > 1 and labels.shape[-1] > 1:
                            labels = torch.argmax(labels, dim=1)
                        labels = labels.to(self.device)

                        logits = self.forward(examples)
                        loss = self.criterion(logits, self.prep_for_loss(labels))
                        test_loss += loss.item()

                test_loss /= len(test_loader)

                # Early stopping
                if test_loss < (min_test_loss - eps):
                    min_test_loss = test_loss
                    patience_counter = 0
                    self.save(checkpoint_path)
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    pbar.set_description(f"Epoch {epoch}, Early stopping triggered.")
                    break
            else:
                test_loss = 0.0

            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, Test: {test_loss:.4f}, Patience: {patience_counter}"
            )

        # Load best model if we had a validation set
        if test_loader is not None:
            self.load(checkpoint_path)

    def predict(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        elif isinstance(X_test, pd.DataFrame):
            X_test = torch.from_numpy(X_test.to_numpy()).float()

        X_test = X_test.to(self.device)
        self.eval()
        with torch.no_grad():
            probs = self.predict_proba(X_test)  # [B, C]
            preds = torch.argmax(probs, dim=1)
            return preds.squeeze().float()

    def predict_proba(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        elif isinstance(X_test, pd.DataFrame):
            X_test = torch.from_numpy(X_test.to_numpy()).float()

        X_test = X_test.to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(X_test)  # [B, target_size]
            probs = self.final_activation(logits)
            # For binary classification => shape [B,1], convert to [B,2]
            if self.target_size == 1:
                probs = torch.hstack([1 - probs, probs])
            return probs.float()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
