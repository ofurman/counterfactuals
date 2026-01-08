import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from counterfactuals.cf_methods.global_methods.globe_ce import GLOBE_CE
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def one_hot(dataset: Any, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply one-hot encoding to categorical features in the dataset.

    Mirrors the AReS preprocessing utility: encodes categoricals and optionally
    bins continuous features, updating dataset metadata for later use.

    Args:
        dataset: Dataset object that will be updated with encoding metadata
        data: DataFrame with raw features to be encoded

    Returns:
        Tuple containing:
            - data_oh: One-hot encoded DataFrame
            - features: List of feature names after encoding
    """
    label_encoder = LabelEncoder()
    data_encode = data.copy()
    dataset.bins = {}
    dataset.bins_tree = {}
    dataset.features_tree = {}
    dataset.n_bins = None

    # Assign encoded features to one hot columns
    data_oh, features = [], []
    for x in data.columns:
        dataset.features_tree[x] = []
        categorical = x in dataset.categorical_features
        if categorical:
            data_encode[x] = label_encoder.fit_transform(data_encode[x])
            cols = label_encoder.classes_
        elif dataset.n_bins is not None:
            data_encode[x] = pd.cut(data_encode[x].apply(lambda x: float(x)), bins=dataset.n_bins)
            cols = data_encode[x].cat.categories
            dataset.bins_tree[x] = {}
        else:
            data_oh.append(data[x])
            features.append(x)
            continue

        one_hot = pd.get_dummies(data_encode[x])
        data_oh.append(one_hot)
        for col in cols:
            feature_value = x + " = " + str(col)
            features.append(feature_value)
            dataset.features_tree[x].append(feature_value)
            if not categorical:
                dataset.bins[feature_value] = col.mid
                dataset.bins_tree[x][feature_value] = col.mid

    data_oh = pd.concat(data_oh, axis=1, ignore_index=True)
    data_oh.columns = features
    return data_oh, features


def compute_bin_widths(
    dataset: MethodDataset, data: pd.DataFrame, n_bins: int = 10
) -> Dict[str, float]:
    """Compute equal-width bin sizes for each continuous feature.

    Args:
        dataset: Dataset carrying feature metadata.
        data: DataFrame containing feature values in original scale.
        n_bins: Number of bins to use for continuous features.

    Returns:
        Mapping from continuous feature name to bin width.
    """
    bin_widths: Dict[str, float] = {}
    for feature in data.columns:
        if feature in dataset.categorical_features:
            continue

        try:
            categories = pd.cut(data[feature].astype(float), bins=n_bins).cat.categories
        except ValueError as err:
            logger.warning(
                "Skipping bin width computation for feature %s: %s", feature, err
            )
            continue

        if len(categories) == 0:
            logger.warning(
                "Skipping bin width computation for feature %s: no categories returned",
                feature,
            )
            continue

        bin_widths[feature] = float(categories.length[-1])

    return bin_widths


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate counterfactuals using the GLOBE-CE method.

    This function constructs required helpers (AReS for bin widths, GLOBE-CE for
    explanation), filters out target-class instances, computes a plausibility
    threshold using the generative model, and generates counterfactuals.

    Args:
        cfg: Hydra configuration with counterfactual parameters
        dataset: Dataset with training/test data and metadata
        gen_model: Pre-trained generative model
        disc_model: Pre-trained discriminative model
        save_folder: Directory path where counterfactuals will be saved

    Returns:
        Tuple containing:
            - Xs_cfs: Generated counterfactual examples
            - Xs: Original examples used for generation
            - log_prob_threshold: Calculated log probability threshold
            - ys_orig: Original predicted labels
            - ys_target: Target labels for counterfactuals
            - model_returned: Boolean mask indicating successful generation
    """
    cf_method_name = "GLOBE_CE"
    disc_model.eval()
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    # Get the MinMaxScaler step directly to avoid torch conversion
    minmax_scaler = dataset.preprocessing_pipeline.get_step("minmax")

    X_test_unscaled = minmax_scaler._inverse_transform_array(dataset.X_test)
    data_oh, features = one_hot(
        dataset, pd.DataFrame(X_test_unscaled, columns=dataset.features)
    )

    def predict_fn(x: pd.DataFrame | np.ndarray) -> np.ndarray:
        # Convert pandas DataFrame to numpy array if needed
        x_array = x.values if isinstance(x, pd.DataFrame) else x
        x_scaled = minmax_scaler._transform_array(x_array)
        return disc_model.predict(x_scaled)

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    ys_pred = predict_fn(X_test_unscaled)
    mask = ys_pred != target_class
    Xs_unscaled = X_test_unscaled[mask]
    Xs = dataset.X_test[mask]
    ys_orig = ys_pred[mask]

    logger.info("Computing bin widths for continuous features")
    bin_widths = compute_bin_widths(
        dataset=dataset,
        data=pd.DataFrame(X_test_unscaled, columns=dataset.features),
        n_bins=10,
    )

    cf_method = GLOBE_CE(
        predict_fn=predict_fn,
        dataset=dataset,
        X=pd.DataFrame(Xs_unscaled, columns=dataset.features),
        bin_widths=bin_widths,
        target_class=target_class,
    )
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")

    logger.info("Handling counterfactual generation")
    time_start = time()
    ys_target = np.full_like(ys_orig, target_class)
    explanation_result = cf_method.explain(
        y_origin=ys_orig,
        y_target=ys_target,
    )
    Xs_cfs = explanation_result.x_cfs
    Xs_cfs = minmax_scaler._transform_array(Xs_cfs)
    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)
    cf_search_time = np.mean(time() - time_start)
    logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info(f"Counterfactuals saved to {counterfactuals_path}")

    return Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target, model_returned


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: List[int],
    continuous_features: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    y_target: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for generated counterfactuals.

    Evaluates the quality of counterfactuals using validity, plausibility,
    proximity, and diversity measures.
    """
    logger.info("Calculating metrics")
    metrics = evaluate_cf(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        median_log_prob=median_log_prob,
        y_target=y_target,
    )
    logger.info(f"Metrics:\n{metrics}")
    return metrics


@hydra.main(config_path="./conf", config_name="globe_ce_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline function for GLOBE-CE counterfactual generation and evaluation.

    Orchestrates a 5-fold CV pipeline: loads dataset, creates discriminative and
    generative models, generates counterfactuals using GLOBE-CE, and evaluates
    them with standard metrics.

    Args:
        cfg: Hydra configuration including dataset, model, and CF parameters
    """
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Loading dataset")
    file_dataset = instantiate(cfg.dataset)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("torch_dtype", TorchDataTypeStep()),
            ("minmax", MinMaxScalingStep()),
        ]
    )
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        logger.info(f"Processing fold {fold_n}")
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        gen_model = create_gen_model(cfg, dataset, gen_model_path)

        Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target, model_returned = search_counterfactuals(
            cfg, dataset, gen_model, disc_model, save_folder
        )

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
        )
        df_metrics = pd.DataFrame(metrics, index=[0])
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )


if __name__ == "__main__":
    main()
