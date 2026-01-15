"""Data utilities for the DiCoFlex counterfactual method."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from counterfactuals.datasets.method_dataset import MethodDataset


def build_actionability_mask(dataset: MethodDataset) -> np.ndarray:
    """Construct a mask that highlights actionable features in the preprocessed space.

    Args:
        dataset: MethodDataset instance with fitted preprocessing pipeline.

    Returns:
        np.ndarray: Vector with shape (n_features,) where actionable positions are set to 1.
    """
    n_features = dataset.X_train.shape[1]
    mask = np.zeros(n_features, dtype=np.float32)
    actionable = set(dataset.actionable_features or [])

    # Numerical features retain a one-to-one mapping after preprocessing.
    for feature_name, feature_idx in zip(
        dataset.numerical_features, dataset.numerical_features_indices
    ):
        if feature_name in actionable:
            mask[feature_idx] = 1.0

    # Categorical features might expand via one-hot encoding. We rely on the helper
    # that exposes one-hot groups and fall back to raw indices if unavailable.
    categorical_groups: Iterable[Sequence[int]]
    try:
        categorical_groups = dataset.categorical_features_lists
    except (AttributeError, ValueError):
        categorical_groups = dataset.categorical_features_indices

    if categorical_groups:
        if (
            isinstance(categorical_groups, list)
            and categorical_groups
            and isinstance(categorical_groups[0], list)
        ):
            cat_groups_iter = categorical_groups
        else:
            cat_groups_iter = [
                [int(index)] if isinstance(index, (int, np.integer)) else [int(i) for i in index]
                for index in categorical_groups
            ]

        for feature_name, indices in zip(dataset.categorical_features, cat_groups_iter):
            if feature_name in actionable:
                mask[np.array(indices, dtype=int)] = 1.0

    if not np.any(mask):
        # Fall back to allowing all features if no actionable metadata is available.
        mask[:] = 1.0
    return mask


@dataclass
class DiCoFlexDatasetConfig:
    """Container describing how to assemble the DiCoFlex training dataset."""

    masks: List[np.ndarray]
    p_values: List[float]
    n_neighbors: int
    noise_level: float
    factual_chunk_size: int | None = None
    target_chunk_size: int | None = None
    seed: int | None = None


class DiCoFlexTrainingDataset(Dataset):
    """Pairs factual points with nearby counterfactual targets for flow training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: DiCoFlexDatasetConfig,
        numerical_indices: Sequence[int],
        categorical_indices: Sequence[int],
    ) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not config.masks:
            raise ValueError("At least one mask vector must be provided.")
        self.X = X.astype(np.float32)
        self.y = y.astype(int).reshape(-1)
        self.n_features = self.X.shape[1]
        self.masks = [self._prepare_mask(mask) for mask in config.masks]
        self.p_values = [float(p) for p in config.p_values]
        self.n_neighbors = max(1, config.n_neighbors)
        self.noise_level = max(0.0, config.noise_level)
        self._rng = np.random.default_rng(config.seed)
        self.numerical_indices = np.array(numerical_indices, dtype=int)
        self.categorical_indices = np.array(categorical_indices, dtype=int)
        self.classes = np.unique(self.y)
        if len(self.classes) < 2:
            raise ValueError("DiCoFlex requires at least two classes.")
        self.class_to_index: Dict[int, int] = {cls: idx for idx, cls in enumerate(self.classes)}
        self.total_candidates = max(
            self.n_neighbors * max(len(self.classes) - 1, 1), self.n_neighbors
        )
        self.factual_chunk_size = (
            max(1, config.factual_chunk_size) if config.factual_chunk_size else 512
        )
        self.target_chunk_size = (
            max(1, config.target_chunk_size) if config.target_chunk_size else 1024
        )
        # Pre-separate data by class to reduce memory usage during neighbor computation
        self._X_by_class: Dict[int, np.ndarray] = {}
        self._indices_by_class: Dict[int, np.ndarray] = {}
        for cls in self.classes:
            class_mask = self.y == cls
            self._X_by_class[cls] = self.X[class_mask]
            self._indices_by_class[cls] = np.where(class_mask)[0]
        self._neighbor_map = self._precompute_neighbors()
        self._factual_entries = self._build_factual_entries()
        if not self._factual_entries:
            raise ValueError(
                "Failed to build DiCoFlex training pairs. "
                "Please check the masks, p-values, or n_neighbors configuration."
            )

    @property
    def context_dim(self) -> int:
        """Return the dimensionality of the conditioning vector."""
        return self.n_features + len(self.classes) + self.n_features + 1

    @property
    def mask_vectors(self) -> List[np.ndarray]:
        """Expose the validated mask vectors."""
        return self.masks

    def __len__(self) -> int:
        return len(self._factual_entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx, p_value, factual_idx, neighbor_records = self._factual_entries[idx]
        factual = self.X[factual_idx]
        cf_samples: List[np.ndarray] = []
        contexts: List[np.ndarray] = []
        for cf_idx, target_class in neighbor_records:
            counterfactual = self._apply_noise(self.X[cf_idx].copy())
            cf_samples.append(counterfactual.astype(np.float32))

            class_one_hot = np.zeros(len(self.classes), dtype=np.float32)
            class_idx = self.class_to_index[target_class]
            class_one_hot[class_idx] = 1.0
            context_vec = np.concatenate(
                [
                    factual,
                    class_one_hot,
                    self.masks[mask_idx],
                    np.array([p_value], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
            contexts.append(context_vec)

        return torch.from_numpy(np.stack(cf_samples)), torch.from_numpy(np.stack(contexts))

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        vector = np.asarray(mask, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.n_features:
            raise ValueError(
                f"Mask length {vector.shape[0]} does not match feature dimension {self.n_features}."
            )
        return vector

    def _precompute_neighbors(self) -> Dict[Tuple[int, float, int, int], np.ndarray]:
        """Compute nearest neighbors using class-separated, chunked distance computation.

        Distances are computed between factual/target class subsets in chunks, reducing
        peak memory usage while still selecting the closest candidates.
        """
        neighbor_map: Dict[Tuple[int, float, int, int], np.ndarray] = {}
        for mask_idx, mask in enumerate(self.masks):
            for p_value in self.p_values:
                for target_class in self.classes:
                    target_indices = np.where(self.y == target_class)[0]
                    if target_indices.size == 0:
                        continue
                    targets = self.X[target_indices]
                    max_neighbors = min(self.total_candidates, target_indices.size)
                    neighbor_ids = np.argsort(distances, axis=1)[:, :max_neighbors]
                    neighbor_map[(mask_idx, p_value, target_class)] = target_indices[neighbor_ids]
        return neighbor_map

    def _compute_neighbors_chunked(
        self,
        X_factual: np.ndarray,
        X_target: np.ndarray,
        target_global_indices: np.ndarray,
        mask: np.ndarray,
        p_value: float,
    ) -> np.ndarray:
        max_neighbors = min(self.total_candidates, X_target.shape[0])
        factual_chunk = min(self.factual_chunk_size, X_factual.shape[0])
        target_chunk = min(self.target_chunk_size, X_target.shape[0])
        neighbor_global_ids = np.empty(
            (X_factual.shape[0], max_neighbors), dtype=target_global_indices.dtype
        )
        mask_weight = mask.reshape(1, 1, -1)

        for start in range(0, X_factual.shape[0], factual_chunk):
            end = min(start + factual_chunk, X_factual.shape[0])
            factual_block = X_factual[start:end]
            best_dists = np.full(
                (factual_block.shape[0], max_neighbors),
                np.inf,
                dtype=np.float32,
            )
            best_indices = np.full(
                (factual_block.shape[0], max_neighbors),
                -1,
                dtype=target_global_indices.dtype,
            )

            for t_start in range(0, X_target.shape[0], target_chunk):
                t_end = min(t_start + target_chunk, X_target.shape[0])
                target_block = X_target[t_start:t_end]
                diff = (
                    np.abs(factual_block[:, None, :] - target_block[None, :, :])
                    ** p_value
                )
                diff *= mask_weight
                distances = np.sum(diff, axis=2) ** (1.0 / p_value)

                target_ids = target_global_indices[t_start:t_end]
                combined_dists = np.concatenate([best_dists, distances], axis=1)
                combined_indices = np.concatenate(
                    [
                        best_indices,
                        np.broadcast_to(target_ids, distances.shape),
                    ],
                    axis=1,
                )
                partition = np.argpartition(combined_dists, max_neighbors - 1, axis=1)[
                    :, :max_neighbors
                ]
                best_dists = np.take_along_axis(combined_dists, partition, axis=1)
                best_indices = np.take_along_axis(combined_indices, partition, axis=1)

            order = np.argsort(best_dists, axis=1)
            neighbor_global_ids[start:end] = np.take_along_axis(
                best_indices, order, axis=1
            )

        return neighbor_global_ids

    def _build_factual_entries(
        self,
    ) -> List[Tuple[int, float, int, List[Tuple[int, int]]]]:
        """Build training entries pairing factual points with their nearest counterfactuals.

        Iterates by class to align with the class-separated neighbor map structure.
        """
        entries: List[Tuple[int, float, int, List[Tuple[int, int]]]] = []

        # Iterate by factual class to match the neighbor map's class-separated structure
        for factual_class in self.classes:
            factual_global_indices = self._indices_by_class[factual_class]

            for local_factual_idx, factual_idx in enumerate(factual_global_indices):
                for mask_idx, _ in enumerate(self.masks):
                    for p_value in self.p_values:
                        neighbor_records: List[Tuple[int, int]] = []

                        for target_class in self.classes:
                            if target_class == factual_class:
                                continue

                            key = (mask_idx, p_value, factual_class, target_class)
                            if key not in self._neighbor_map:
                                continue

                            # Use local index to access the neighbor map
                            neighbor_indices = self._neighbor_map[key][
                                local_factual_idx
                            ]
                            neighbor_records.extend(
                                [(cf_idx, target_class) for cf_idx in neighbor_indices]
                            )

                        if not neighbor_records:
                            continue

                        if len(neighbor_records) < self.n_neighbors:
                            repeats = math.ceil(
                                self.n_neighbors / len(neighbor_records)
                            )
                            neighbor_records = (neighbor_records * repeats)[
                                : self.n_neighbors
                            ]
                        else:
                            neighbor_records = neighbor_records[: self.n_neighbors]

                        entries.append(
                            (mask_idx, p_value, factual_idx, neighbor_records)
                        )
                    if not neighbor_records:
                        continue
                    if len(neighbor_records) < self.n_neighbors:
                        repeats = math.ceil(self.n_neighbors / len(neighbor_records))
                        neighbor_records = (neighbor_records * repeats)[: self.n_neighbors]
                    else:
                        neighbor_records = neighbor_records[: self.n_neighbors]
                    entries.append((mask_idx, p_value, factual_idx, neighbor_records))
        return entries

    def _apply_noise(self, sample: np.ndarray) -> np.ndarray:
        if self.noise_level > 0 and self.numerical_indices.size > 0:
            sample[self.numerical_indices] += self._rng.normal(
                0.0,
                self.noise_level,
                size=self.numerical_indices.size,
            ).astype(np.float32)
        if self.noise_level > 0 and self.categorical_indices.size > 0:
            sample[self.categorical_indices] += self._rng.normal(
                0.0,
                self.noise_level * 0.1,
                size=self.categorical_indices.size,
            ).astype(np.float32)
        return sample


def create_dicoflex_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    masks: List[np.ndarray],
    p_values: List[float],
    n_neighbors: int,
    noise_level: float,
    factual_batch_size: int,
    val_ratio: float,
    seed: int,
    numerical_indices: Sequence[int],
    categorical_indices: Sequence[int],
    factual_chunk_size: int | None = None,
    target_chunk_size: int | None = None,
) -> Tuple[DataLoader, DataLoader, Dict[int, int], List[np.ndarray], int]:
    """Create train/validation loaders for DiCoFlex flow training.

    Args:
        factual_chunk_size: Chunk size for factual samples in neighbor search.
        target_chunk_size: Chunk size for target samples in neighbor search.
    """
    config = DiCoFlexDatasetConfig(
        masks=masks,
        p_values=p_values,
        n_neighbors=n_neighbors,
        noise_level=noise_level,
        factual_chunk_size=factual_chunk_size,
        target_chunk_size=target_chunk_size,
        seed=seed,
    )
    dataset = DiCoFlexTrainingDataset(
        X=X,
        y=y,
        config=config,
        numerical_indices=numerical_indices,
        categorical_indices=categorical_indices,
    )
    if len(dataset) < 2:
        raise ValueError("Not enough training pairs to train DiCoFlex flow.")
    val_size = max(1, int(len(dataset) * val_ratio))
    if val_size >= len(dataset):
        val_size = max(1, len(dataset) // 5)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for the available DiCoFlex training pairs.")
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_subset,
        batch_size=factual_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=factual_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return (
        train_loader,
        val_loader,
        dataset.class_to_index,
        dataset.mask_vectors,
        dataset.context_dim,
    )
