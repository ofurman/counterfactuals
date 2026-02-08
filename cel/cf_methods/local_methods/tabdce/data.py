from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TabularSpec:
    """Specification of numerical and categorical feature indices."""

    num_idx: list[int]
    cat_idx: list[int]
    cat_cardinalities: Optional[list[int]] = None


class TabularCounterfactualDataset(Dataset):
    """Dataset wrapper for TabDCE diffusion training and transforms."""

    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        spec: TabularSpec,
        k: int = 15,
        search_method: Literal["knn", "dpp"] = "knn",
        dpp_pool_factor: int = 3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        qt: QuantileTransformer | None = None,
        ohe: OneHotEncoder | None = None,
        build_neighbors: bool = True,
    ) -> None:
        """Initialize the dataset and precompute model-space inputs.

        Args:
            X: Feature matrix.
            y: Labels.
            spec: Feature specification with numerical/categorical indices.
            k: Number of opposite-class neighbors to sample.
            search_method: Strategy for neighbor selection (knn or dpp).
            dpp_pool_factor: Pool multiplier used by the DPP sampler.
            device: Torch device for tensors.
            dtype: Torch dtype for model-space features.
            qt: Optional quantile transformer for numerical features.
            ohe: Optional one-hot encoder for categorical features.
            build_neighbors: Whether to precompute neighbors for training.
        """
        super().__init__()

        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.spec = spec
        self.k = k
        self.search_method = search_method
        self.dpp_pool_factor = dpp_pool_factor

        if isinstance(y, torch.Tensor):
            self.y = y.long().to(self.device)
        else:
            self.y = torch.from_numpy(y).long().to(self.device)

        self.num_classes_target = len(torch.unique(self.y))
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        num_samples = X_np.shape[0]

        if self.spec.num_idx:
            X_num_np = X_np[:, self.spec.num_idx].astype(np.float32)
        else:
            X_num_np = np.zeros((num_samples, 0), dtype=np.float32)

        if self.spec.cat_idx:
            X_cat_np = X_np[:, self.spec.cat_idx]
        else:
            X_cat_np = np.zeros((num_samples, 0))

        self.num_numerical = X_num_np.shape[1]
        if self.num_numerical > 0:
            if qt is None:
                self.qt = QuantileTransformer(
                    output_distribution="normal", n_quantiles=min(1000, num_samples)
                )
                X_num_tr = self.qt.fit_transform(X_num_np)
            else:
                self.qt = qt
                X_num_tr = self.qt.transform(X_num_np)
        else:
            self.qt = None
            X_num_tr = np.zeros((num_samples, 0), dtype=np.float32)

        if X_cat_np.shape[1] > 0:
            if ohe is None:
                self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                X_cat_oh = self.ohe.fit_transform(X_cat_np)
            else:
                self.ohe = ohe
                X_cat_oh = self.ohe.transform(X_cat_np)
            self.cat_cardinalities = [len(cats) for cats in self.ohe.categories_]
            X_cat_log = np.log(np.clip(X_cat_oh, 1e-30, 1.0)).astype(np.float32)
        else:
            self.ohe = None
            self.cat_cardinalities = []
            X_cat_log = np.zeros((num_samples, 0), dtype=np.float32)

        if self.spec.cat_cardinalities is None:
            self.spec.cat_cardinalities = self.cat_cardinalities

        X_model_np = np.concatenate([X_num_tr.astype(np.float32), X_cat_log], axis=1)
        self.X_model = torch.from_numpy(X_model_np).to(self.device).to(dtype)

        if build_neighbors:
            if self.num_numerical > 0:
                X_feat = torch.from_numpy(X_num_tr.astype(np.float32)).to(self.device)
            else:
                X_feat = torch.zeros((num_samples, 1), device=self.device)
            logger.info(
                "Building neighbors using method: %s", self.search_method.upper()
            )
            self.neigh_idx = self._build_opposite_class_neighbors(
                X_feat, self.y, k=self.k
            )
        else:
            self.neigh_idx = None

    def _build_opposite_class_neighbors(
        self, X_feat: torch.Tensor, y: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Build neighbor indices from opposite-class samples."""
        num_samples = X_feat.size(0)
        neigh_all = torch.zeros((num_samples, k), dtype=torch.long, device=self.device)
        classes = y.unique().tolist()

        k_pool = k * self.dpp_pool_factor if self.search_method == "dpp" else k

        with torch.no_grad():
            for cls in classes:
                src_idx = (y == cls).nonzero(as_tuple=False).squeeze(1)
                tgt_idx = (y != cls).nonzero(as_tuple=False).squeeze(1)

                if src_idx.numel() == 0:
                    continue
                if tgt_idx.numel() == 0:
                    rand = torch.randint(
                        0, num_samples, (src_idx.numel(), k), device=self.device
                    )
                    neigh_all[src_idx] = rand
                    continue

                A = X_feat[src_idx]
                B = X_feat[tgt_idx]
                curr_pool = min(k_pool, int(tgt_idx.size(0)))
                dists = torch.cdist(A, B)
                topk_res = torch.topk(dists, k=curr_pool, largest=False)

                candidates_local = topk_res.indices
                candidates_global = tgt_idx[candidates_local]

                if self.search_method == "knn":
                    chosen = candidates_global[:, :k]
                else:
                    chosen = self._select_dpp_greedy(
                        query_feats=A,
                        cand_feats=X_feat[candidates_global],
                        cand_indices=candidates_global,
                        k=k,
                    )

                if chosen.size(1) < k:
                    pad_size = k - chosen.size(1)
                    pad = chosen[:, -1:].repeat(1, pad_size)
                    chosen = torch.cat([chosen, pad], dim=1)

                neigh_all[src_idx] = chosen

        return neigh_all

    def _select_dpp_greedy(
        self,
        query_feats: torch.Tensor,
        cand_feats: torch.Tensor,
        cand_indices: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Greedy k-DPP selection for diverse neighbor sampling."""
        batch_size, pool_size, _ = cand_feats.shape
        if pool_size <= k:
            return cand_indices

        sigma_q = 1.0
        sigma_s = 5.0

        selected_indices = torch.zeros(
            (batch_size, k), dtype=torch.long, device=self.device
        )

        for i in range(batch_size):
            q_vec = query_feats[i].unsqueeze(0)
            c_mat = cand_feats[i]
            c_idx = cand_indices[i]

            norm_c = (c_mat**2).sum(dim=1, keepdim=True)
            dist_cc = norm_c + norm_c.t() - 2 * (c_mat @ c_mat.t())
            S = torch.exp(-dist_cc / sigma_s)
            S = S + torch.eye(pool_size, device=self.device) * 1e-4

            dist_qc = torch.cdist(q_vec, c_mat).squeeze(0) ** 2
            Q = torch.exp(-dist_qc / sigma_q)
            L = S * torch.outer(Q, Q)
            cis: list[int] = []

            available = list(range(pool_size))

            for step in range(k):
                if step == 0:
                    best_local_idx = torch.argmax(torch.diag(L)).item()
                else:
                    best_gain = -float("inf")
                    best_local_idx = -1

                    for idx in available:
                        temp_cis = cis + [idx]
                        L_sub = L[np.ix_(temp_cis, temp_cis)]
                        gain = torch.logdet(L_sub)
                        if gain > best_gain:
                            best_gain = gain
                            best_local_idx = idx

                if best_local_idx in available:
                    cis.append(best_local_idx)
                    available.remove(best_local_idx)
                else:
                    break

            selected_indices[i] = c_idx[cis]

        return selected_indices

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.X_model.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a training pair for the diffusion model."""
        x_orig = self.X_model[idx]
        y_orig = self.y[idx]
        if self.neigh_idx is not None:
            cand_indices = self.neigh_idx[idx]
            chosen_idx = cand_indices[
                torch.randint(0, cand_indices.numel(), (1,)).item()
            ]
            x_neigh = self.X_model[chosen_idx]
            y_tgt = self.y[chosen_idx]
        else:
            x_neigh = x_orig.clone()
            y_tgt = y_orig
        return {"x_orig": x_orig, "y": y_orig, "x_neigh": x_neigh, "y_target": y_tgt}

    def to_model_space(self, X_raw: np.ndarray) -> torch.Tensor:
        """Transform raw data into the diffusion model feature space."""
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
        return torch.from_numpy(X_model).to(self.device).to(self.dtype)

    def inverse_transform(self, x_model: torch.Tensor | np.ndarray) -> np.ndarray:
        """Invert model-space features back to the original tabular space."""
        if isinstance(x_model, torch.Tensor):
            x_model = x_model.detach().cpu().numpy()
        num_samples = x_model.shape[0]

        if self.num_numerical > 0:
            x_num_tr = x_model[:, : self.num_numerical]
            x_num_tr = np.clip(x_num_tr, -5.2, 5.2)
            x_num_orig = (
                self.qt.inverse_transform(x_num_tr) if self.qt is not None else x_num_tr
            )
        else:
            x_num_orig = np.zeros((num_samples, 0))

        if self.ohe is not None:
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
            x_cat_orig = np.hstack(x_cat_orig_list)
        else:
            x_cat_orig = np.zeros((num_samples, 0))

        return np.concatenate([x_num_orig, x_cat_orig], axis=1)
