import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from counterfactuals.cf_methods.base import BaseCounterfactualModel
from counterfactuals.sparsemax import Sparsemax


class PPCEF_2(torch.nn.Module):
    def __init__(self, N, D, K):
        super(PPCEF_2, self).__init__()
        assert K == N, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.d = torch.nn.Parameter(torch.zeros((N, D)))

    def forward(self):
        return self.d

    def get_matrices(self):
        return torch.ones(self.N, 1), torch.ones(self.N, self.K), self.d

    def loss(self, *args, **kwargs):
        return torch.Tensor([0])


class ARES(torch.nn.Module):
    def __init__(self, N, D, K=1):
        super(ARES, self).__init__()
        assert K == 1, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.d = torch.nn.Parameter(torch.zeros(self.K, self.D))

    def forward(self):
        return torch.ones(self.N, self.K) @ self.d

    def get_matrices(self):
        return torch.ones(self.N, 1), torch.ones(self.N, self.K), self.d

    def loss(self, *args, **kwargs):
        return torch.Tensor([0])


class GLOBAL_CE(torch.nn.Module):
    def __init__(self, N, D, K):
        super(GLOBAL_CE, self).__init__()
        assert K == 1, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.m = torch.nn.Parameter(torch.zeros(self.N, 1))
        self.d = torch.nn.Parameter(torch.zeros((self.K, self.D)))

    def forward(self):
        return torch.exp(self.m) @ self.d

    def get_matrices(self):
        return torch.exp(self.m), torch.ones(self.N, self.K), self.d

    def loss(self, *args, **kwargs):
        return torch.Tensor([0])


class GCE(torch.nn.Module):
    def __init__(self, N, D, K):
        super(GCE, self).__init__()
        assert 1 <= K and K <= N, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.m = torch.nn.Parameter(0 * torch.rand(self.N, 1))
        self.s = torch.nn.Parameter(0.01 * torch.rand(self.N, self.K))
        self.d = torch.nn.Parameter(0 * torch.rand((self.K, self.D)))

        self.sparsemax = Sparsemax(dim=1)

    def _entropy_loss(self, prob_dist):
        prob_dist = torch.clamp(prob_dist, min=1e-9)
        row_wise_entropy = -torch.sum(prob_dist * torch.log(prob_dist), dim=1)
        return row_wise_entropy

    def forward(self):
        return torch.exp(self.m) * self.sparsemax(self.s) @ self.d

    def rows_entropy(self):
        row_wise_entropy = self._entropy_loss(self.sparsemax(self.s))
        return row_wise_entropy

    def cols_entropy(self):
        s_col_prob = self.sparsemax(self.s).sum(axis=0) / self.sparsemax(self.s).sum()
        s_col_prob = s_col_prob.clamp(min=1e-9)
        col_wise_entropy = -torch.sum(s_col_prob * torch.log(s_col_prob))
        return col_wise_entropy

    def loss(self, alpha_s, alpha_k):
        return alpha_s * self.rows_entropy() + alpha_k * self.cols_entropy()

    def get_matrices(self):
        return torch.exp(self.m), self.sparsemax(self.s), self.d


class RPPCEF(BaseCounterfactualModel):
    def __init__(
        self,
        delta,
        gen_model,
        disc_model,
        disc_model_criterion,
        device=None,
        neptune_run=None,
    ):
        self.delta = delta
        self.disc_model_criterion = disc_model_criterion
        super().__init__(gen_model, disc_model, device, neptune_run)

    def search_step(
            self, delta, x_origin, contexts_origin, context_target, **search_step_kwargs
    ) -> dict:
        """Search step for the cf search process.
        :param x_param: point to be optimized
        :param x_origin: original point
        :param context_target: target context
        :param search_step_kwargs: dict with additional parameters
        :return: dict with loss and additional components to log.
        """
        alpha = search_step_kwargs.get("alpha", None)
        alpha_s = search_step_kwargs.get("alpha_s", None)
        alpha_k = search_step_kwargs.get("alpha_k", None)
        median_log_prob = search_step_kwargs.get("median_log_prob", None)

        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if alpha_plausability is None:
            alpha_plausability = alpha
        if median_log_prob is None:
            raise ValueError("Parameter 'median_log_prob' should be in kwargs")

        dist = torch.linalg.vector_norm(delta(), dim=1, ord=2)

        disc_logits = self.disc_model.forward(x_origin + delta())
        disc_logits = (
            disc_logits.reshape(-1) if disc_logits.shape[0] == 1 else disc_logits
        )
        context_target = (
            context_target.reshape(-1).float()
            if context_target.shape[0] == 1
            else context_target.long()
        )
        loss_disc = self.disc_model_criterion(disc_logits, context_target)

        p_x_param_c_target = self.gen_model(
            x_origin + delta(), context=context_target.type(torch.float32)
        ).clamp(max=10 ** 3)
        max_inner = torch.nn.functional.relu(median_log_prob - p_x_param_c_target)

        delta_loss = delta.loss(alpha_s, alpha_k)

        # loss = dist + alpha * (loss_disc + max_inner) # + sparse_loss + col_wise_entropy)
        loss = dist + alpha * (loss_disc + max_inner) + delta_loss
        # loss = dist + alpha * (max_inner + loss_disc) + alpha/10 * (sparse_loss + col_wise_entropy)

        return {
            "loss": loss,
            "dist": dist,
            "max_inner": max_inner,
            "loss_disc": loss_disc,
            "delta_loss": delta_loss,
        }

    def search_batch(
            self,
            dataloader: DataLoader,
            epochs: int = 1000,
            lr: float = 0.0005,
            patience: int = 100,
            patience_eps: int = 1e-3,
            **search_step_kwargs,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

        if self.disc_model:
            self.disc_model.eval()
            for param in self.disc_model.parameters():
                param.requires_grad = False

        deltas = []
        target_class = []
        original = []
        original_class = []
        for xs_origin, contexts_origin in dataloader:
            xs_origin = xs_origin.to(self.device)
            contexts_origin = contexts_origin.to(self.device)

            contexts_origin = contexts_origin.reshape(-1, 1)
            contexts_target = torch.abs(1 - contexts_origin)

            xs_origin = torch.as_tensor(xs_origin)
            xs_origin.requires_grad = False

            optimizer = torch.optim.Adam(self.delta.parameters(), lr=lr)

            loss_components_logging = {}
            min_loss = float("inf")

            for epoch in (epoch_pbar := tqdm(range(epochs), dynamic_ncols=True)):
                optimizer.zero_grad()
                loss_components = self.search_step(
                    self.delta,
                    xs_origin,
                    contexts_origin,
                    contexts_target,
                    **search_step_kwargs,
                )
                mean_loss = loss_components["loss"].mean()
                mean_loss.backward()
                optimizer.step()

                for loss_name, loss in loss_components.items():
                    loss_components_logging.setdefault(
                        f"cf_search/{loss_name}", []
                    ).append(loss.mean().detach().cpu().item())
                    if self.neptune_run:
                        self.neptune_run[f"cf_search/{loss_name}"].append(
                            loss.mean().detach().cpu().numpy()
                        )

                loss = loss_components["loss"].detach().cpu().mean().item()
                epoch_pbar.set_description(
                    ", ".join(
                        [
                            f"{k}: {v.detach().cpu().mean().item():.4f}"
                            for k, v in loss_components.items()
                        ]
                    )
                )
                if (loss < (min_loss - patience_eps)) or (epoch < 1000):
                    min_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        break

            deltas.append(self.delta)
            original.append(xs_origin.detach().cpu().numpy())
            original_class.append(contexts_origin.detach().cpu().numpy())
            target_class.append(contexts_target.detach().cpu().numpy())
        return (
            deltas,
            np.concatenate(original, axis=0),
            np.concatenate(original_class, axis=0),
            np.concatenate(target_class, axis=0),
            loss_components_logging,
        )
