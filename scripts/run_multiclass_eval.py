"""Evaluate CF methods across all K*(K-1) directed class pairs and aggregate.

Trains models from scratch for the 3-class blobs dataset, then evaluates
DICE, CCHVAE (pairwise), and DiCoFlex on all 6 directed class pairs.
A single CF model per method is trained once and reused for every target class.

Usage:
    uv run python scripts/run_multiclass_eval.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cel.cf_methods.local_methods.c_chvae.c_chvae import CCHVAE
from cel.cf_methods.local_methods.c_chvae.data import CustomData
from cel.cf_methods.local_methods.c_chvae.mlmodel import CustomMLModel
from cel.cf_methods.local_methods.dicoflex import DiCoFlex, DiCoFlexParams
from cel.cf_methods.local_methods.dicoflex.context_utils import (
    DiCoFlexGeneratorMetricsAdapter,
    build_context_matrix,
    get_numpy_pointer,
)
from cel.cf_methods.local_methods.dicoflex.data import (
    build_actionability_mask,
    create_dicoflex_dataloaders,
)
from cel.datasets.method_dataset import MethodDataset
from cel.metrics.metrics import evaluate_cf
from cel.models.classifier.multilayer_perceptron import MLPClassifier
from cel.models.generative.maf.maf import MaskedAutoregressiveFlow
from cel.pipelines.runners.dicoflex_runner import (
    compute_log_prob_threshold,
    get_full_training_loader,
    train_dicoflex_generator,
)
from cel.preprocessing import MinMaxScalingStep, PreprocessingPipeline, TorchDataTypeStep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_CONFIG_PATH = "config/datasets/blobs_3class.yaml"
DATASET_NAME = "blobs_3class"
MODELS_ROOT = "models"
METRICS_CONF_PATH = "cel/pipelines/conf/metrics/dicoflex.yaml"
NUM_FOLDS = 5
SEED = 42

# Disc model config
DISC_MODEL_NAME = "MLPClassifier"
DISC_HIDDEN = [256, 256]
DISC_DROPOUT = 0.2
DISC_EPOCHS = 2000
DISC_BATCH_SIZE = 128
DISC_PATIENCE = 200
DISC_LR = 0.001

# DiCoFlex config
DICOFLEX_FLOW_EPOCHS = 200
DICOFLEX_FLOW_PATIENCE = 100
DICOFLEX_FLOW_LR = 0.001
DICOFLEX_P_VALUES = [2.0]
DICOFLEX_N_NEIGHBORS = 16
DICOFLEX_NOISE = 0.02
DICOFLEX_TRAIN_BATCH = 4
DICOFLEX_VAL_RATIO = 0.2
DICOFLEX_NUM_CFS = 32
DICOFLEX_CF_PER_FACTUAL = 20
DICOFLEX_SAMPLING_BS = 256
DICOFLEX_LOG_PROB_Q = 0.25

# CCHVAE config
CCHVAE_HYPERPARAMS: dict = {
    "data_name": "blobs_3class",
    "n_search_samples": 300,
    "p_norm": 1,
    "step": 0.1,
    "max_iter": 2000,
    "clamp": True,
    "binary_cat_features": True,
    "vae_params": {
        "layers": [64, 32, 16],
        "train": True,
        "kl_weight": 0.3,
        "lambda_reg": 1e-6,
        "epochs": 10,
        "lr": 1e-3,
        "batch_size": 32,
    },
}
CCHVAE_NUM_CFS = 20

# DICE config
DICE_TOTAL_CFS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class DummyGenModel(nn.Module):
    """Placeholder gen model for metrics that don't require density estimation."""

    def eval(self) -> DummyGenModel:
        return self

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(x.shape[0])

    def predict_log_prob(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        return torch.zeros(1)


def load_dataset() -> MethodDataset:
    """Load 3-class blobs dataset with standard preprocessing."""
    from cel.datasets import FileDataset

    file_dataset = FileDataset(config_path=DATASET_CONFIG_PATH)
    preprocessing = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    return MethodDataset(file_dataset, preprocessing)


def get_unique_classes(dataset: MethodDataset) -> list[int]:
    """Discover unique classes from train + test labels."""
    all_y = np.concatenate([dataset.y_train, dataset.y_test])
    return sorted(np.unique(all_y).astype(int).tolist())


def fold_dir(fold: int) -> str:
    """Return the model directory for a given fold."""
    return os.path.join(MODELS_ROOT, DATASET_NAME, f"fold_{fold}")


def disc_model_path(fold: int) -> str:
    return os.path.join(fold_dir(fold), f"disc_model_{DISC_MODEL_NAME}.pt")


def dicoflex_gen_model_path(fold: int) -> str:
    return os.path.join(
        fold_dir(fold),
        f"gen_model_MaskedAutoregressiveFlow_relabeled_by_{DISC_MODEL_NAME}.pt",
    )


def filter_test_data(dataset: MethodDataset, target_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Filter test data to exclude target class instances."""
    mask = dataset.y_test != target_class
    return dataset.X_test[mask], dataset.y_test[mask]


@dataclass
class DirectionResult:
    """Metrics for a single (fold, target_class) evaluation."""

    fold: int
    target_class: int
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_or_load_disc_model(dataset: MethodDataset, fold: int) -> nn.Module:
    """Train disc model if checkpoint missing, otherwise load it."""
    path = disc_model_path(fold)
    num_classes = len(np.unique(dataset.y_train))
    model = MLPClassifier(
        num_inputs=dataset.X_train.shape[1],
        num_targets=num_classes,
        hidden_layer_sizes=DISC_HIDDEN,
        dropout=DISC_DROPOUT,
    )
    if os.path.exists(path):
        logger.info("Loading disc model from %s", path)
        model.load(path)
    else:
        logger.info("Training disc model for fold %d (%d classes)", fold, num_classes)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        train_dl = dataset.train_dataloader(batch_size=DISC_BATCH_SIZE, shuffle=True, noise_lvl=0)
        test_dl = dataset.test_dataloader(batch_size=DISC_BATCH_SIZE, shuffle=False)
        model.fit(
            train_dl,
            test_dl,
            epochs=DISC_EPOCHS,
            lr=DISC_LR,
            patience=DISC_PATIENCE,
            checkpoint_path=path,
        )
        model.save(path)
    model.eval()
    return model


def relabel_with_disc_model(dataset: MethodDataset, disc_model: nn.Module) -> None:
    """Replace dataset labels with disc model predictions (in-place)."""
    dataset.y_train = disc_model.predict(dataset.X_train)
    dataset.y_test = disc_model.predict(dataset.X_test)


# ---------------------------------------------------------------------------
# DiCoFlex: train flow once per fold, reuse for all target classes
# ---------------------------------------------------------------------------
@dataclass
class DiCoFlexFoldState:
    """Pre-computed DiCoFlex state for a single fold (shared across targets)."""

    gen_model: nn.Module
    class_to_index: dict
    mask_vectors: list[np.ndarray]
    context_dim: int
    log_prob_threshold: float


def prepare_dicoflex_fold(dataset: MethodDataset, fold: int) -> DiCoFlexFoldState:
    """Train or load the DiCoFlex conditional flow for one fold."""
    device = "cpu"
    torch.manual_seed(SEED)

    masks: list[np.ndarray] = [build_actionability_mask(dataset)]
    if not masks or masks[0] is None:
        masks = [np.ones(dataset.X_train.shape[1], dtype=np.float32)]

    train_loader, val_loader, class_to_index, mask_vectors, context_dim = (
        create_dicoflex_dataloaders(
            dataset.X_train,
            dataset.y_train,
            masks=masks,
            p_values=DICOFLEX_P_VALUES,
            n_neighbors=DICOFLEX_N_NEIGHBORS,
            noise_level=DICOFLEX_NOISE,
            factual_batch_size=DICOFLEX_TRAIN_BATCH,
            val_ratio=DICOFLEX_VAL_RATIO,
            seed=SEED,
            numerical_indices=dataset.numerical_features_indices,
            categorical_indices=dataset.categorical_features_indices,
        )
    )

    gen_model = MaskedAutoregressiveFlow(
        features=dataset.X_train.shape[1],
        context_features=context_dim,
        hidden_features=16,
        num_blocks_per_layer=4,
        num_layers=8,
    ).to(device)

    path = dicoflex_gen_model_path(fold)
    if os.path.exists(path):
        try:
            gen_model.load(path)
            logger.info("Loaded DiCoFlex gen model from %s", path)
        except RuntimeError:
            logger.info("Saved model incompatible, retraining DiCoFlex gen model for fold %d", fold)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cfg = OmegaConf.create(
                {
                    "gen_model": {
                        "epochs": DICOFLEX_FLOW_EPOCHS,
                        "patience": DICOFLEX_FLOW_PATIENCE,
                        "lr": DICOFLEX_FLOW_LR,
                    },
                }
            )
            train_dicoflex_generator(gen_model, train_loader, val_loader, cfg, path, device)
    else:
        logger.info("Training DiCoFlex gen model for fold %d", fold)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cfg = OmegaConf.create(
            {
                "gen_model": {
                    "epochs": DICOFLEX_FLOW_EPOCHS,
                    "patience": DICOFLEX_FLOW_PATIENCE,
                    "lr": DICOFLEX_FLOW_LR,
                },
            }
        )
        train_dicoflex_generator(gen_model, train_loader, val_loader, cfg, path, device)

    full_loader = get_full_training_loader(train_loader, DICOFLEX_TRAIN_BATCH)
    log_prob_threshold = compute_log_prob_threshold(
        gen_model, full_loader, DICOFLEX_LOG_PROB_Q, device
    )

    return DiCoFlexFoldState(
        gen_model=gen_model,
        class_to_index=class_to_index,
        mask_vectors=mask_vectors,
        context_dim=context_dim,
        log_prob_threshold=log_prob_threshold,
    )


# ---------------------------------------------------------------------------
# DICE evaluation
# ---------------------------------------------------------------------------
def run_dice_direction(
    dataset: MethodDataset,
    disc_model: nn.Module,
    target_class: int,
) -> dict[str, float]:
    """Run DiCE for one target_class direction, return metrics dict."""
    import dice_ml

    num_classes = len(np.unique(dataset.y_train))
    X_test_orig, y_test_orig = filter_test_data(dataset, target_class)
    X_test_orig = X_test_orig.astype(np.float64)
    y_test_orig = y_test_orig.astype(np.float64)

    features = [str(i) for i in range(dataset.X_train.shape[1])] + ["label"]
    X_combined = np.concatenate([dataset.X_train, X_test_orig], axis=0)
    y_combined = np.concatenate([dataset.y_train, y_test_orig], axis=0)
    combined_df = pd.DataFrame(
        np.concatenate((X_combined, y_combined.reshape(-1, 1)), axis=1),
        columns=features,
    )

    dice_data = dice_ml.Data(
        dataframe=combined_df,
        continuous_features=[str(i) for i in range(dataset.X_train.shape[1])],
        outcome_name=features[-1],
    )

    class _DiscWrapper(nn.Module):
        def __init__(self, model: nn.Module, n_classes: int) -> None:
            super().__init__()
            self.model = model
            self.n_classes = n_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.model(x)
            if self.n_classes <= 2 and logits.shape[-1] == 1:
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=-1)

    dice_model = dice_ml.Model(_DiscWrapper(disc_model, num_classes), backend="PYT")
    exp = dice_ml.Dice(dice_data, dice_model, method="random")
    query = pd.DataFrame(X_test_orig, columns=features[:-1])

    t0 = time()
    cfs = exp.generate_counterfactuals(
        query,
        total_CFs=DICE_TOTAL_CFS,
        desired_class=int(target_class),
        posthoc_sparsity_param=None,
    )
    cf_time = time() - t0

    Xs_cfs, Xs_origs, ys_origs, group_ids = [], [], [], []
    for idx, (orig, y_orig_i, cf) in enumerate(zip(X_test_orig, y_test_orig, cfs.cf_examples_list)):
        if cf.final_cfs_df is None or cf.final_cfs_df.shape[0] == 0:
            Xs_cfs.append(orig)
            Xs_origs.append(orig)
            ys_origs.append(y_orig_i)
            group_ids.append(idx)
            continue
        out = cf.final_cfs_df.to_numpy()[:, :-1]
        n = out.shape[0]
        Xs_cfs.append(out)
        Xs_origs.append(np.tile(orig, (n, 1)))
        ys_origs.append(np.full(n, y_orig_i))
        group_ids.append(np.full(n, idx))

    X_cf = np.concatenate([np.atleast_2d(x) for x in Xs_cfs], axis=0)
    X_test_flat = np.concatenate([np.atleast_2d(x) for x in Xs_origs], axis=0)
    y_orig_flat = np.concatenate([np.atleast_1d(y) for y in ys_origs], axis=0)
    cf_group_ids = np.concatenate([np.atleast_1d(g) for g in group_ids], axis=0)
    y_target_flat = np.full_like(y_orig_flat, fill_value=target_class, dtype=y_orig_flat.dtype)
    model_returned = np.ones(X_cf.shape[0], dtype=bool)

    gen_model = DummyGenModel()
    metrics = evaluate_cf(
        disc_model=disc_model,
        gen_model=gen_model,
        X_cf=X_cf,
        model_returned=model_returned,
        continuous_features=dataset.numerical_features_indices,
        categorical_features=dataset.categorical_features_indices,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=X_test_flat,
        y_test=y_orig_flat,
        y_target=y_target_flat,
        median_log_prob=0.0,
        cf_group_ids=cf_group_ids,
        metrics_conf_path=METRICS_CONF_PATH,
    )
    metrics["cf_search_time"] = cf_time
    return metrics


# ---------------------------------------------------------------------------
# CCHVAE evaluation (pairwise)
# ---------------------------------------------------------------------------
def run_cchvae_direction(
    dataset: MethodDataset,
    disc_model: nn.Module,
    target_class: int,
) -> dict[str, float]:
    """Run CCHVAE pairwise for one target_class direction, return metrics dict."""
    X_test_orig, y_test_orig = filter_test_data(dataset, target_class)

    custom_dataset = CustomData(dataset)
    wrapped_model = CustomMLModel(disc_model, custom_dataset)

    hyperparams = dict(CCHVAE_HYPERPARAMS)
    input_size = dataset.X_train.shape[1]
    hyperparams["vae_params"] = dict(hyperparams["vae_params"])
    hyperparams["vae_params"]["layers"] = [input_size] + hyperparams["vae_params"]["layers"]

    exp = CCHVAE(wrapped_model, hyperparams)

    cf_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test_orig).float(),
            torch.tensor(y_test_orig).float(),
        ),
        batch_size=4096,
        shuffle=False,
    )

    y_target = np.full_like(y_test_orig, fill_value=target_class)

    t0 = time()
    cfs_list: list[np.ndarray] = []
    for _ in range(CCHVAE_NUM_CFS):
        result = exp.explain_dataloader(
            dataloader=cf_dataloader,
            epochs=20000,
            lr=0.001,
            y_target=y_target,
        )
        cfs_list.append(result.x_cfs)
    cf_time = time() - t0

    Xs_cfs_all = np.stack(cfs_list, axis=1)
    n_instances, n_runs, n_features = Xs_cfs_all.shape
    X_cf_flat = Xs_cfs_all.reshape(-1, n_features)
    X_test_flat = np.repeat(X_test_orig, n_runs, axis=0)
    y_orig_flat = np.repeat(y_test_orig, n_runs, axis=0)
    y_target_flat = np.repeat(y_target, n_runs, axis=0)
    cf_group_ids = np.repeat(np.arange(n_instances), n_runs)
    model_returned = np.ones(X_cf_flat.shape[0], dtype=bool)

    gen_model = DummyGenModel()
    metrics = evaluate_cf(
        disc_model=disc_model,
        gen_model=gen_model,
        X_cf=X_cf_flat,
        model_returned=model_returned,
        continuous_features=dataset.numerical_features_indices,
        categorical_features=dataset.categorical_features_indices,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=X_test_flat,
        y_test=y_orig_flat,
        y_target=y_target_flat,
        median_log_prob=0.0,
        cf_group_ids=cf_group_ids,
        metrics_conf_path=METRICS_CONF_PATH,
    )
    metrics["cf_search_time"] = cf_time
    return metrics


# ---------------------------------------------------------------------------
# DiCoFlex evaluation (uses pre-computed fold state)
# ---------------------------------------------------------------------------
def run_dicoflex_direction(
    dataset: MethodDataset,
    disc_model: nn.Module,
    target_class: int,
    fold_state: DiCoFlexFoldState,
) -> dict[str, float]:
    """Run DiCoFlex for one target_class direction using pre-trained flow."""
    device = "cpu"

    params = DiCoFlexParams(
        mask_index=0,
        p_value=DICOFLEX_P_VALUES[0],
        num_counterfactuals=DICOFLEX_NUM_CFS,
        target_class=target_class,
        sampling_batch_size=DICOFLEX_SAMPLING_BS,
        cf_samples_per_factual=DICOFLEX_CF_PER_FACTUAL,
    )
    cf_method = DiCoFlex(
        gen_model=fold_state.gen_model,
        disc_model=disc_model,
        class_to_index=fold_state.class_to_index,
        mask_vectors=fold_state.mask_vectors,
        params=params,
        device=device,
    )

    X_test_orig, y_test_orig = filter_test_data(dataset, target_class)
    cf_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_test_orig).float(),
            torch.from_numpy(y_test_orig).float(),
        ),
        batch_size=DICOFLEX_SAMPLING_BS,
        shuffle=False,
    )

    t0 = time()
    explanation = cf_method.explain_dataloader(cf_loader, epochs=0, lr=0.0)
    cf_time = time() - t0

    X_cf = np.ascontiguousarray(explanation.x_cfs.astype(np.float32, copy=False))
    X_origs = np.ascontiguousarray(explanation.x_origs.astype(np.float32, copy=False))
    cf_group_ids = (
        None
        if explanation.cf_group_ids is None
        else np.asarray(explanation.cf_group_ids, dtype=int)
    )
    model_returned_mask = np.array(explanation.logs.get("model_returned_mask", []), dtype=bool)
    if model_returned_mask.size == 0:
        model_returned_mask = np.ones(X_cf.shape[0], dtype=bool)

    mask_vector = fold_state.mask_vectors[0]
    cf_contexts = build_context_matrix(
        factual_points=X_origs,
        labels=explanation.y_cf_targets,
        mask_vector=mask_vector,
        p_value=DICOFLEX_P_VALUES[0],
        class_to_index=fold_state.class_to_index,
    )
    test_contexts = build_context_matrix(
        factual_points=X_origs,
        labels=explanation.y_origs,
        mask_vector=mask_vector,
        p_value=DICOFLEX_P_VALUES[0],
        class_to_index=fold_state.class_to_index,
    )
    context_lookup = {
        get_numpy_pointer(X_cf): cf_contexts,
        get_numpy_pointer(X_origs): test_contexts,
    }
    metrics_gen_model = DiCoFlexGeneratorMetricsAdapter(
        base_model=fold_state.gen_model, context_lookup=context_lookup
    )

    metrics = evaluate_cf(
        disc_model=disc_model,
        gen_model=metrics_gen_model,
        X_cf=X_cf,
        model_returned=model_returned_mask,
        continuous_features=dataset.numerical_features_indices,
        categorical_features=dataset.categorical_features_indices,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=X_origs,
        y_test=explanation.y_origs,
        y_target=explanation.y_cf_targets,
        median_log_prob=fold_state.log_prob_threshold,
        cf_group_ids=cf_group_ids,
        metrics_conf_path=METRICS_CONF_PATH,
    )
    metrics["cf_search_time"] = cf_time
    return metrics


# ---------------------------------------------------------------------------
# Aggregation & display
# ---------------------------------------------------------------------------
METRIC_DISPLAY = {
    "validity": "Val.",
    "proximity_l1_continuous": "Prox.-Cont",
    "eps_sparsity": "eps-Spars.",
    "sparsity_categorical": "Spars.-Cat",
    "pairwise_diversity_mixed": "Diversity",
    "lof_score_median_log": "LOF",
}


def aggregate_results(results: list[DirectionResult]) -> pd.Series:
    """Compute instance-weighted average across all (fold, target_class) runs."""
    rows = [{"fold": r.fold, "target_class": r.target_class, **r.metrics} for r in results]
    df = pd.DataFrame(rows)

    if "number_of_instances" not in df.columns or df["number_of_instances"].sum() == 0:
        return df.drop(columns=["fold", "target_class"], errors="ignore").mean()

    weights = df["number_of_instances"]
    total = weights.sum()
    agg: dict[str, float] = {}
    for col in df.columns:
        if col in ("fold", "target_class"):
            continue
        if col == "number_of_instances":
            agg[col] = total
        elif col == "cf_search_time":
            agg[col] = df[col].mean()
        else:
            agg[col] = float((df[col] * weights).sum() / total)
    return pd.Series(agg)


def print_comparison_table(method_results: dict[str, pd.Series], classes: list[int]) -> None:
    """Print a comparison table."""
    methods = list(method_results.keys())
    col_w = max(12, max(len(n) for n in METRIC_DISPLAY.values()) + 2)
    method_w = max(len(m) for m in methods) + 2

    hdr = f"  {'Metric':<{col_w}}"
    for m in methods:
        hdr += f" | {m:^{method_w}}"
    border = "  " + "-" * col_w
    for _ in methods:
        border += "-+-" + "-" * method_w

    n_pairs = len(classes) * (len(classes) - 1)
    print("\n" + "=" * 70)
    print("MULTI-CLASS PAIRWISE EVALUATION (blobs_3class)")
    print(f"Classes: {classes} | Pairs: {n_pairs} | Folds: {NUM_FOLDS}")
    print("=" * 70)
    print(border)
    print(hdr)
    print(border)
    for metric_key, display_name in METRIC_DISPLAY.items():
        row = f"  {display_name:<{col_w}}"
        for m in methods:
            val = method_results[m].get(metric_key, float("nan"))
            row += f" | {val:^{method_w}.3f}"
        print(row)
    print(border)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.manual_seed(SEED)

    dataset = load_dataset()
    classes = get_unique_classes(dataset)
    n_pairs = len(classes) * (len(classes) - 1)
    logger.info(
        "Dataset: %s | Classes: %s | Pairs: %d | Folds: %d",
        DATASET_NAME,
        classes,
        n_pairs,
        NUM_FOLDS,
    )

    method_aggregated: dict[str, pd.Series] = {}

    for method_name in ["DICE", "CCHVAE", "DiCoFlex"]:
        logger.info("=" * 60)
        logger.info("Method: %s", method_name)
        logger.info("=" * 60)

        all_results: list[DirectionResult] = []

        for fold_n, _ in enumerate(dataset.get_cv_splits(NUM_FOLDS)):
            # Train / load disc model (shared across methods within fold)
            disc_model = train_or_load_disc_model(dataset, fold_n)
            relabel_with_disc_model(dataset, disc_model)

            # Pre-compute DiCoFlex fold state (train flow once per fold)
            dicoflex_state: DiCoFlexFoldState | None = None
            if method_name == "DiCoFlex":
                dicoflex_state = prepare_dicoflex_fold(dataset, fold_n)

            for target_class in classes:
                logger.info("  Fold %d | target=%d | %s", fold_n, target_class, method_name)
                try:
                    if method_name == "DiCoFlex":
                        metrics = run_dicoflex_direction(
                            dataset, disc_model, target_class, dicoflex_state
                        )
                    elif method_name == "DICE":
                        metrics = run_dice_direction(dataset, disc_model, target_class)
                    else:
                        metrics = run_cchvae_direction(dataset, disc_model, target_class)

                    all_results.append(
                        DirectionResult(
                            fold=fold_n,
                            target_class=target_class,
                            metrics=metrics,
                        )
                    )
                    logger.info(
                        "    val=%.3f  prox=%.3f  n=%d",
                        metrics.get("validity", float("nan")),
                        metrics.get("proximity_l1_continuous", float("nan")),
                        int(metrics.get("number_of_instances", 0)),
                    )
                except Exception:
                    logger.exception("    FAILED fold=%d target=%d", fold_n, target_class)

        if all_results:
            agg = aggregate_results(all_results)
            method_aggregated[method_name] = agg
            logger.info("Aggregated %s: %s", method_name, agg.to_dict())
        else:
            logger.warning("No results for %s", method_name)

    print_comparison_table(method_aggregated, classes)

    if method_aggregated:
        df_out = pd.DataFrame(method_aggregated).T
        df_out.index.name = "method"
        out_path = "outputs/blobs_3class_results.csv"
        os.makedirs("outputs", exist_ok=True)
        df_out.to_csv(out_path)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
