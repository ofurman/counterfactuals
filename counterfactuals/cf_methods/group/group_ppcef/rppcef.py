import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from counterfactuals.cf_methods.base import BaseCounterfactual
from counterfactuals.generative_models.base import BaseGenModel
from counterfactuals.discriminative_models.base import BaseDiscModel
from counterfactuals.cf_methods.group_ppcef.deltas import PPCEF_2, ARES, GLOBAL_CE, GCE


class RPPCEF(BaseCounterfactual):
    def __init__(
        self,
        cf_method_type: str,
        gen_model: BaseGenModel,
        disc_model: BaseDiscModel,
        disc_model_criterion: torch.nn.modules.loss._Loss,
        init_cf_method_from_kmeans: bool = False,
        K: int = None,
        X: np.ndarray = None,
        device: str = None,
        neptune_run=None,
    ):
        self.delta = self._init_cf_method(
            cf_method_type, K, init_cf_method_from_kmeans, X
        )
        self.disc_model_criterion = disc_model_criterion
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.device = device if device else "cpu"
        self.neptune_run = neptune_run
        self.loss_components_logs = {}

    def _init_cf_method(
        self,
        cf_method_type: str,
        K: int,
        init_cf_method_from_kmeans: bool = False,
        X=None,
    ):
        N = X.shape[0]
        D = X.shape[1]
        if cf_method_type in ["ARES", "GLOBAL_CE"]:
            K = 1
        elif cf_method_type == "PPCEF_2":
            K = N
        elif K is not None:
            K = K
        elif X is not None:
            K = X.shape[0]
        else:
            raise ValueError("K or X must be provided")

        cf_methods = {
            "ARES": ARES,
            "GLOBAL_CE": GLOBAL_CE,
            "GCE": GCE,
            "PPCEF_2": PPCEF_2,
        }
        if cf_method_type not in cf_methods:
            raise ValueError(f"Unknown cf_method: {cf_method_type}")

        if cf_method_type == "GCE":
            return cf_methods[cf_method_type](N, D, K, init_cf_method_from_kmeans, X)
        return cf_methods[cf_method_type](N, D, K)

    def _search_step(
        self,
        delta,
        x_origin,
        contexts_origin,
        context_target,
        alpha,
        alpha_s,
        alpha_k,
        log_prob_threshold,
    ) -> dict:
        """Search step for the cf search process.
        :param x_param: point to be optimized
        :param x_origin: original point
        :param context_target: target context
        :param search_step_kwargs: dict with additional parameters
        :param alpha: weight for the loss
        :param alpha_s: weight for the loss_disc
        :param alpha_k: weight for the delta loss
        :param log_prob_threshold: threshold for the log probability
        :return: dict with loss and additional components to log.
        """
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

    def _log_loss_components(self, loss_components):
        for loss_name, loss in loss_components.items():
            self.loss_components_logs.setdefault(f"cf_search/{loss_name}", []).append(
                loss.mean().detach().cpu().item()
            )
            if self.neptune_run:
                self.neptune_run[f"cf_search/{loss_name}"].append(
                    loss.mean().detach().cpu().numpy()
                )

    def explain_dataloader(
        self,
        dataloader: DataLoader,
        alpha: int,
        alpha_s: int,
        alpha_k: int,
        log_prob_threshold: float,
        epochs: int = 1000,
        lr: float = 0.0005,
        patience: int = 100,
        patience_eps: int = 1e-3,
    ):
        """
        Trains the model for a specified number of epochs.
        """
        self.loss_components_logs = {}
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

            min_loss = float("inf")
            dist_flag = False

            for epoch in (epoch_pbar := tqdm(range(epochs), dynamic_ncols=True)):
                optimizer.zero_grad()
                loss_components = self._search_step(
                    self.delta,
                    xs_origin,
                    contexts_origin,
                    contexts_target,
                    alpha=alpha,
                    alpha_s=alpha_s,
                    alpha_k=alpha_k,
                    log_prob_threshold=log_prob_threshold,
                )
                mean_loss = loss_components["loss"].mean()
                mean_loss.backward()
                optimizer.step()

                self._log_loss_components(loss_components)

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

        x_origs = np.concatenate(original, axis=0)
        y_origs = np.concatenate(original_class, axis=0)
        y_target = np.concatenate(target_class, axis=0)
        # x_cfs = x_origs + self.delta().detach().numpy()
        return self.delta, x_origs, y_origs, y_target
