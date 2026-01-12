from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from counterfactuals.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)
from counterfactuals.cf_methods.local_methods.tabdce.data import TabularSpec
from counterfactuals.cf_methods.local_methods.tabdce.diffusion import (
    MixedTabularDiffusion,
)


class TabDCE(BaseCounterfactualMethod, LocalCounterfactualMixin):
    """TabDCE counterfactual generator for tabular data."""

    def __init__(
        self,
        diffusion_model: MixedTabularDiffusion,
        spec: TabularSpec,
        qt: Optional[QuantileTransformer],
        ohe: Optional[OneHotEncoder],
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=str(device))
        self.diffusion_model = diffusion_model.to(device)
        self.spec = spec
        self.qt = qt
        self.ohe = ohe
        self.device = torch.device(device)
        self.num_numerical = len(spec.num_idx)
        self.cat_cardinalities = (
            [len(cats) for cats in ohe.categories_] if ohe is not None else []
        )

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        batch_size: int = 512,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactuals for a batch of instances."""
        _ = X_train, y_train, kwargs
        X_tensor = torch.tensor(X).float()
        y_origin_tensor = torch.tensor(y_origin).long()
        y_target_tensor = torch.tensor(y_target).long()
        dataloader = DataLoader(
            TensorDataset(X_tensor, y_origin_tensor, y_target_tensor),
            batch_size=batch_size,
            shuffle=False,
        )
        return self.explain_dataloader(dataloader=dataloader)

    def explain_dataloader(
        self,
        dataloader: DataLoader,
        target_class: Optional[int] = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactuals for a DataLoader of inputs."""
        _ = kwargs
        self.diffusion_model.eval()

        x_cfs: list[np.ndarray] = []
        x_origs: list[np.ndarray] = []
        y_origs: list[np.ndarray] = []
        y_targets: list[np.ndarray] = []

        for batch in dataloader:
            if isinstance(batch, dict):
                x_batch = batch.get("x_orig", batch.get("x"))
                y_batch = batch.get("y")
                y_target = batch.get("y_target")
            else:
                x_batch = batch[0]
                y_batch = batch[1] if len(batch) > 1 else None
                y_target = batch[2] if len(batch) > 2 else None

            if x_batch is None:
                raise ValueError(
                    "Dataloader must yield input features as the first element."
                )

            x_np = x_batch.detach().cpu().numpy()
            x_model = self._to_model_space(x_np)

            if y_target is None:
                if target_class is None:
                    raise ValueError(
                        "Provide target_class or per-sample y_target values."
                    )
                y_target_tensor = torch.full(
                    (x_model.size(0),),
                    int(target_class),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                y_target_tensor = y_target.to(self.device).long()

            with torch.no_grad():
                x_cf_model = self.diffusion_model.sample_counterfactual(
                    x_orig=x_model.to(self.device),
                    y_target=y_target_tensor,
                )

            x_cf = self._inverse_transform(x_cf_model)
            x_cfs.append(x_cf)
            x_origs.append(x_np)
            if y_batch is None:
                y_origs.append(np.zeros((x_np.shape[0],), dtype=int))
            else:
                y_origs.append(y_batch.detach().cpu().numpy())
            y_targets.append(y_target_tensor.detach().cpu().numpy())

        return ExplanationResult(
            x_cfs=np.concatenate(x_cfs, axis=0) if x_cfs else np.empty((0,)),
            y_cf_targets=np.concatenate(y_targets, axis=0)
            if y_targets
            else np.empty((0,)),
            x_origs=np.concatenate(x_origs, axis=0) if x_origs else np.empty((0,)),
            y_origs=np.concatenate(y_origs, axis=0) if y_origs else np.empty((0,)),
        )

    def _to_model_space(self, X_raw: np.ndarray) -> torch.Tensor:
        """Transform raw features into the diffusion model space."""
        num_samples = X_raw.shape[0]
        if self.num_numerical > 0:
            X_num = X_raw[:, self.spec.num_idx].astype(np.float32)
            X_num_tr = self.qt.transform(X_num) if self.qt is not None else X_num
        else:
            X_num_tr = np.zeros((num_samples, 0), dtype=np.float32)

        if self.ohe is not None:
            X_cat = X_raw[:, self.spec.cat_idx]
            X_cat_oh = self.ohe.transform(X_cat)
            X_cat_log = np.log(np.clip(X_cat_oh, 1e-30, 1.0)).astype(np.float32)
        else:
            X_cat_log = np.zeros((num_samples, 0), dtype=np.float32)

        X_model = np.concatenate([X_num_tr, X_cat_log], axis=1)
        return torch.from_numpy(X_model).to(self.device)

    def _inverse_transform(self, x_model: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert model-space features to the original tabular space."""
        if isinstance(x_model, torch.Tensor):
            x_model = x_model.detach().cpu().numpy()
        num_samples = x_model.shape[0]

        if self.num_numerical > 0:
            x_num_tr = x_model[:, : self.num_numerical]
            x_num_tr = np.clip(x_num_tr, -5.2, 5.2)
            x_num = (
                self.qt.inverse_transform(x_num_tr) if self.qt is not None else x_num_tr
            )
        else:
            x_num = np.zeros((num_samples, 0), dtype=np.float32)

        if self.ohe is not None and self.cat_cardinalities:
            x_cat_part = x_model[:, self.num_numerical :]
            indices_list = []
            start = 0
            for k in self.cat_cardinalities:
                part = x_cat_part[:, start : start + k]
                idx = np.argmax(part, axis=1)
                indices_list.append(idx.reshape(-1, 1))
                start += k
            cat_indices = np.hstack(indices_list)
            x_cat_orig_list = []
            for i, cats in enumerate(self.ohe.categories_):
                col_indices = cat_indices[:, i]
                orig_vals = cats[col_indices]
                x_cat_orig_list.append(orig_vals.reshape(-1, 1))
            x_cat = np.hstack(x_cat_orig_list)
        else:
            x_cat = np.zeros((num_samples, 0), dtype=np.float32)

        return np.concatenate([x_num, x_cat], axis=1)
