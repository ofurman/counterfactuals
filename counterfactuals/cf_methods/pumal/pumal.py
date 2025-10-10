import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from counterfactuals.cf_methods.base import BaseCounterfactual
from counterfactuals.cf_methods.pumal.deltas import (
    ARES,
    GCE,
    GLOBAL_CE,
    PPCEF_2,
    DimConfig,
    GradStrategy,
)
from counterfactuals.discriminative_models.base import BaseDiscModel
from counterfactuals.generative_models.base import BaseGenModel


class PUMAL(BaseCounterfactual):
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
        not_actionable_features: list = None,
        increase_only_features: list = None,
        decrease_only_features: list = None,
        feature_ranges: dict = None,
        neptune_run=None,
    ):
        """
        Initialize the PUMAL counterfactual method.

        Args:
            cf_method_type: Type of counterfactual method to use ('ARES', 'GLOBAL_CE', 'GCE', 'PPCEF_2')
            gen_model: Generative model
            disc_model: Discriminative model
            disc_model_criterion: Loss function for the discriminative model
            init_cf_method_from_kmeans: Whether to initialize from KMeans
            K: Number of clusters
            X: Training data
            device: Device to use (cpu/cuda)
            not_actionable_features: List of feature indices that should not be modified
            increase_only_features: List of feature indices that can only increase in value
            decrease_only_features: List of feature indices that can only decrease in value
            feature_ranges: Dictionary mapping feature indices to (min, max) tuples for clamping
            neptune_run: Neptune run for logging
        """
        self.not_actionable_features = not_actionable_features
        self.increase_only_features = increase_only_features
        self.decrease_only_features = decrease_only_features
        self.feature_ranges = feature_ranges

        self.delta = self._init_cf_method(
            cf_method_type, K, init_cf_method_from_kmeans, X
        )
        self.disc_model_criterion = disc_model_criterion
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.device = device if device else "cpu"
        self.neptune_run = neptune_run
        self.loss_components_logs = {}

    def _prepare_dim_configs(self, D):
        """
        Prepare dimension configurations for the GCE model.

        Args:
            D: Dimensionality of the data

        Returns:
            Dictionary mapping dimension indices to DimConfig objects
        """
        dim_configs = {}

        # Setup not actionable features (zero gradient)
        if self.not_actionable_features:
            for dim in self.not_actionable_features:
                dim_configs[dim] = DimConfig(strategy=GradStrategy.ZERO)

        # Setup increase only features
        if self.increase_only_features:
            for dim in self.increase_only_features:
                # If this dimension already has a config, only update its strategy
                if dim in dim_configs:
                    dim_configs[dim].strategy = GradStrategy.INCREASE_ONLY
                else:
                    dim_configs[dim] = DimConfig(strategy=GradStrategy.INCREASE_ONLY)

        # Setup decrease only features
        if self.decrease_only_features:
            for dim in self.decrease_only_features:
                # If this dimension already has a config, only update its strategy
                if dim in dim_configs:
                    dim_configs[dim].strategy = GradStrategy.DECREASE_ONLY
                else:
                    dim_configs[dim] = DimConfig(strategy=GradStrategy.DECREASE_ONLY)

        # Setup feature ranges
        if self.feature_ranges:
            for dim, (min_val, max_val) in self.feature_ranges.items():
                # If this dimension already has a config, update its range
                if dim in dim_configs:
                    dim_configs[dim].min_val = min_val
                    dim_configs[dim].max_val = max_val
                else:
                    dim_configs[dim] = DimConfig(min_val=min_val, max_val=max_val)

        return dim_configs

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
        elif cf_method_type == "GCE" and K is not None:
            K = K
        elif cf_method_type == "GCE" and K is None:
            K = N
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
            # Prepare dimension configurations
            dim_configs = self._prepare_dim_configs(D)
            print(dim_configs)

            return cf_methods[cf_method_type](
                N, D, K, init_cf_method_from_kmeans, X, dim_configs=dim_configs
            )
        return cf_methods[cf_method_type](N, D, K)

    def _search_step(
        self,
        delta,
        x_origin,
        contexts_origin,
        context_target,
        alpha_dist,
        alpha_plaus,
        alpha_class,
        alpha_d,
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
        dist = alpha_dist * torch.linalg.vector_norm(delta(), dim=1, ord=1)

        disc_logits = self.disc_model(x_origin + delta())
        disc_logits = (
            disc_logits.reshape(-1) if disc_logits.shape[0] == 1 else disc_logits
        )
        # context_target = (
        #     context_target.reshape(-1).float()
        #     if context_target.shape[0] == 1
        #     else context_target.long()
        # )
        loss_disc = alpha_class * self.disc_model_criterion(disc_logits, context_target)

        p_x_param_c_target = self.gen_model(
            x_origin + delta(), context=context_target.float()
        ).clamp(max=10**5)
        max_inner = alpha_plaus * torch.nn.functional.relu(
            log_prob_threshold - p_x_param_c_target
        )

        delta_loss = delta.loss(alpha_s, alpha_k, alpha_d)
        loss = alpha_dist * dist + loss_disc + max_inner + delta_loss

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
        target_class: int,
        log_prob_threshold: float,
        alpha_dist: float = 1e-1,
        alpha_plaus: float = 10**4,
        alpha_class: float = 10**5,
        alpha_s: float = 10**4,
        alpha_k: float = 10**3,
        alpha_d: float = 10**2,
        epochs: int = 1000,
        lr: float = 0.0005,
        patience: int = 100,
        patience_eps: int = 1e-3,
        decrease_loss_patience: int = 500,
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

        target_classes = []
        original = []
        original_classes = []
        for xs_origin, contexts_origin in dataloader:
            xs_origin = xs_origin.to(self.device)
            contexts_origin = contexts_origin.to(self.device)

            if len(contexts_origin.shape) == 1:
                contexts_origin = contexts_origin.reshape(-1, 1)
            # contexts_origin = contexts_origin.reshape(-1, 10)
            contexts_target = torch.zeros_like(contexts_origin)
            print(contexts_target.shape)
            contexts_target[:, target_class] = 1

            xs_origin = torch.as_tensor(xs_origin)
            xs_origin.requires_grad = False

            optimizer = torch.optim.Adam(self.delta.parameters(), lr=lr)
            # scheduler = MultiStepLR(
            #     optimizer, milestones=[decrease_loss_after_steps], gamma=0.1
            # )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=decrease_loss_patience
            )

            min_loss = float("inf")
            dist_flag = False

            for epoch in (epoch_pbar := tqdm(range(epochs), dynamic_ncols=True)):
                optimizer.zero_grad()
                loss_components = self._search_step(
                    self.delta,
                    xs_origin,
                    contexts_origin,
                    contexts_target,
                    alpha_dist=alpha_dist,
                    alpha_plaus=alpha_plaus,
                    alpha_class=alpha_class,
                    alpha_s=alpha_s,
                    alpha_k=alpha_k,
                    alpha_d=alpha_d,
                    log_prob_threshold=log_prob_threshold,
                )
                mean_loss = loss_components["loss"].mean()
                mean_loss.backward()
                optimizer.step()

                self._log_loss_components(loss_components)

                loss = loss_components["loss"].detach().cpu().mean().item()
                scheduler.step(loss)
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
            original_classes.append(contexts_origin.detach().cpu().numpy())
            target_classes.append(contexts_target.detach().cpu().numpy())

        x_origs = np.concatenate(original, axis=0)
        y_origs = np.concatenate(original_classes, axis=0)
        y_target = np.concatenate(target_classes, axis=0)
        # x_cfs = x_origs + self.delta().detach().numpy()
        return self.delta, x_origs, y_origs, y_target
