import numpy as np
from sklearn.cluster import KMeans


import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from counterfactuals.cf_methods.ppcef_base import BasePPCEF
from counterfactuals.generative_models.base import BaseGenModel
from counterfactuals.discriminative_models.base import BaseDiscModel
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
    def __init__(self, N, D, K, init_from_kmeans=False, X=None):
        super(GCE, self).__init__()
        assert 1 <= K and K <= N, "Assumption of the method!"
        assert N >= 1
        assert D >= 1

        self.N = N
        self.D = D
        self.K = K

        self.m = torch.nn.Parameter(0 * torch.rand(self.N, 1))
        self.d = torch.nn.Parameter(0 * torch.rand((self.K, self.D)))

        if init_from_kmeans:
            assert X is not None, "X should be provided for KMeans initialization"
            self.s = self._init_from_kmeans(X, K)
        else:
            self.s = torch.nn.Parameter(0.01 * torch.rand(self.N, self.K))
        self.sparsemax = Sparsemax(dim=1)

    def _init_from_kmeans(self, X, K):
        kmeans = KMeans(n_clusters=K, random_state=42).fit(X)
        group_labels = kmeans.labels_
        group_labels_one_hot = np.zeros((group_labels.size, group_labels.max() + 1))
        group_labels_one_hot[np.arange(group_labels.size), group_labels] = 1
        assert group_labels_one_hot.shape[1] == K
        assert group_labels_one_hot.shape[0] == X.shape[0]
        return torch.from_numpy(group_labels_one_hot).float()

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


class RPPCEF(BasePPCEF):
    def __init__(
        self,
        cf_method_type: str,
        N: int,
        D: int,
        K: int,
        gen_model: BaseGenModel,
        disc_model: BaseDiscModel,
        disc_model_criterion: torch.nn.modules.loss._Loss,
        init_cf_method_from_kmeans: bool = False,
        X=None,
        device=None,
        neptune_run=None,
    ):
        self.delta = self._init_cf_method(
            cf_method_type, N, D, K, init_cf_method_from_kmeans, X
        )
        self.disc_model_criterion = disc_model_criterion
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.device = device if device else "cpu"
        self.neptune_run = neptune_run

    def _init_cf_method(
        self, cf_method_type, N, D, K, init_cf_method_from_kmeans=False, X=None
    ):
        if cf_method_type == "ARES":
            return ARES(N, D, K)
        elif cf_method_type == "GLOBAL_CE":
            return GLOBAL_CE(N, D, K)
        elif cf_method_type == "GCE":
            return GCE(N, D, K, init_cf_method_from_kmeans, X)
        elif cf_method_type == "PPCEF_2":
            return PPCEF_2(N, D, K)
        else:
            raise ValueError(f"Unknown cf_method: {cf_method_type}")

    def _search_step(
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
        log_prob_threshold = search_step_kwargs.get("log_prob_threshold", None)
        dist_flag = search_step_kwargs.get("dist_flag", None)

        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if alpha_s is None:
            raise ValueError("Parameter 'alpha_s' should be in kwargs")
        if alpha_k is None:
            raise ValueError("Parameter 'alpha_k' should be in kwargs")
        if log_prob_threshold is None:
            raise ValueError("Parameter 'log_prob_threshold' should be in kwargs")
        if dist_flag is None:
            raise ValueError("Parameter 'dist_flag' should be in kwargs")

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
        ).clamp(max=10**5)
        max_inner = torch.nn.functional.relu(log_prob_threshold - p_x_param_c_target)

        delta_loss = delta.loss(alpha_s, alpha_k)
        # dist = dist if dist_flag else torch.Tensor([0])
        loss = dist + alpha * (loss_disc + max_inner) + delta_loss

        return {
            "loss": loss,
            "dist": dist,
            "max_inner": max_inner,
            "loss_disc": loss_disc,
            "delta_loss": delta_loss,
        }

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """
        Explains the model's prediction for a given input.
        """
        raise NotImplementedError("This method is not implemented for this class.")

    def explain_dataloader(
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
            dist_flag = False

            for epoch in (epoch_pbar := tqdm(range(epochs), dynamic_ncols=True)):
                optimizer.zero_grad()
                loss_components = self._search_step(
                    self.delta,
                    xs_origin,
                    contexts_origin,
                    contexts_target,
                    dist_flag=dist_flag,
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
                # Progress bar description
                epoch_pbar.set_description(
                    ", ".join(
                        [
                            f"{k}: {v.detach().cpu().mean().item():.4f}"
                            for k, v in loss_components.items()
                        ]
                    )
                )
                # Early stopping handling
                if (loss < (min_loss - patience_eps)) or (epoch < 1000):
                    min_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        if not dist_flag:
                            patience_counter = 0
                            dist_flag = True
                        else:
                            break

            original.append(xs_origin.detach().cpu().numpy())
            original_class.append(contexts_origin.detach().cpu().numpy())
            target_class.append(contexts_target.detach().cpu().numpy())
        return (
            self.delta,
            np.concatenate(original, axis=0),
            np.concatenate(original_class, axis=0),
            np.concatenate(target_class, axis=0),
            loss_components_logging,
        )
