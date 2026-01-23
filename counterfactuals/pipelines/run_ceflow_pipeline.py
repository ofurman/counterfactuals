import inspect
import logging
import os
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.cf_methods.local_methods.ceflow.ceflow import CeFlow, CeFlowParams
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.dequantization.utils import DequantizationWrapper
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.pipelines.utils import apply_categorical_discretization
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _resolve_flow_transforms(
    flow_model: torch.nn.Module,
) -> Tuple[
    Optional[Callable[[torch.Tensor], torch.Tensor]],
    Optional[Callable[[torch.Tensor], torch.Tensor]],
]:
    if hasattr(flow_model, "transform_to_latent") and hasattr(
        flow_model, "transform_to_data"
    ):

        def encode(x: torch.Tensor) -> torch.Tensor:
            return flow_model.transform_to_latent(x)

        def decode(z: torch.Tensor) -> torch.Tensor:
            return flow_model.transform_to_data(z)

        return encode, decode

    if hasattr(flow_model, "inverse"):
        return None, None

    flow_core = getattr(flow_model, "model", None)
    if flow_core is None:
        raise ValueError(
            "CeFlow requires a flow model with an inverse or a .model attribute."
        )

    if hasattr(flow_core, "transform_to_noise") and hasattr(
        flow_core, "transform_to_data"
    ):

        def encode(x: torch.Tensor) -> torch.Tensor:
            z_value, _ = flow_core.transform_to_noise(x)
            return z_value

        def decode(z: torch.Tensor) -> torch.Tensor:
            return flow_core.transform_to_data(z)

        return encode, decode

    transform = getattr(flow_core, "_transform", None)
    if transform is None:
        raise ValueError("CeFlow could not find flow transforms on the provided model.")

    def encode(x: torch.Tensor) -> torch.Tensor:
        output = transform.forward(x) if hasattr(transform, "forward") else transform(x)
        return output[0] if isinstance(output, tuple) else output

    def decode(z: torch.Tensor) -> torch.Tensor:
        output = transform.inverse(z) if hasattr(transform, "inverse") else transform(z)
        return output[0] if isinstance(output, tuple) else output

    return encode, decode


def _wrap_with_dequantizer(
    base_encode: Optional[Callable[[torch.Tensor], torch.Tensor]],
    base_decode: Optional[Callable[[torch.Tensor], torch.Tensor]],
    flow_model: torch.nn.Module,
    dequantizer: GroupDequantizer,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
]:
    def encode(x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        x_dq = dequantizer.transform(x_np)
        x_tensor = torch.from_numpy(x_dq).float().to(x.device)
        if base_encode is not None:
            return base_encode(x_tensor)
        return flow_model(x_tensor)

    def decode(z: torch.Tensor) -> torch.Tensor:
        if base_decode is not None:
            x_tensor = base_decode(z)
        else:
            x_tensor = flow_model.inverse(z)
        x_np = x_tensor.detach().cpu().numpy()
        x_inv = dequantizer.inverse_transform(x_np)
        return torch.from_numpy(x_inv).float().to(z.device)

    return encode, decode


def _build_ceflow_params(cfg: DictConfig) -> CeFlowParams:
    return CeFlowParams(
        batch_size=cfg.counterfactuals_params.batch_size,
        alpha_min=cfg.counterfactuals_params.alpha_min,
        alpha_max=cfg.counterfactuals_params.alpha_max,
        alpha_steps=cfg.counterfactuals_params.alpha_steps,
        alpha_grid=list(cfg.counterfactuals_params.alpha_grid),
        distance_metric=cfg.counterfactuals_params.distance_metric,
        binary_logits=cfg.counterfactuals_params.binary_logits,
        clamp_min=cfg.counterfactuals_params.clamp_min,
        clamp_max=cfg.counterfactuals_params.clamp_max,
        use_predicted_labels=cfg.counterfactuals_params.use_predicted_labels,
    )


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    flow_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    save_folder: str,
    dequantizer: GroupDequantizer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate counterfactuals using the CeFlow method.

    Args:
        cfg: Hydra configuration containing counterfactual parameters
        dataset: Dataset containing training and test data
        flow_model: Trained flow model used for latent optimization
        disc_model: Discriminative model to explain
        save_folder: Directory path where counterfactuals will be saved

    Returns:
        tuple: Generated counterfactuals, originals, labels, and runtime info.
    """
    cf_method_name = "CeFlow"
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]

    logger.info("Creating CeFlow counterfactual model")
    if getattr(flow_model, "context_features", None):
        raise ValueError(
            "CeFlow flow_model must be unconditional; set flow_model.context_features to null."
        )
    base_encode, base_decode = _resolve_flow_transforms(flow_model)
    if hasattr(flow_model, "transform_to_latent") and hasattr(
        flow_model, "transform_to_data"
    ):
        encode_fn, decode_fn = base_encode, base_decode
    else:
        encode_fn, decode_fn = _wrap_with_dequantizer(
            base_encode, base_decode, flow_model, dequantizer
        )
    params = _build_ceflow_params(cfg)
    cf_method = CeFlow(
        flow_model=flow_model,
        disc_model=disc_model,
        params=params,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
    )

    logger.info("Handling counterfactual generation")
    time_start = time()
    y_target = np.full_like(y_test_origin, fill_value=target_class)
    explanation_result = cf_method.explain(
        X=X_test_origin,
        y_origin=y_test_origin,
        y_target=y_target,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
    )

    cf_search_time = time() - time_start
    logger.info(f"Counterfactual search time: {cf_search_time:.4f} seconds")

    Xs = explanation_result.x_origs
    Xs_cfs = explanation_result.x_cfs
    ys_orig = explanation_result.y_origs
    ys_target = explanation_result.y_cf_targets
    preds = disc_model.predict(Xs_cfs)
    model_returned = preds.reshape(-1) == ys_target.reshape(-1)

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    if cfg.counterfactuals_params.use_categorical:
        Xs_cfs = apply_categorical_discretization(
            dataset.categorical_features_lists, Xs_cfs
        )

    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    logger.info(f"Counterfactuals saved to: {counterfactuals_path}")

    return Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time


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

    Args:
        gen_model: Generative model used for plausibility assessment
        disc_model: Discriminative model used for validity assessment
        Xs_cfs: Generated counterfactual examples
        model_returned: Boolean array indicating successful counterfactual generation
        categorical_features: List of categorical feature indices
        continuous_features: List of continuous feature indices
        X_train: Training data features
        y_train: Training data labels
        X_test: Original test examples
        y_test: Original test labels
        median_log_prob: Log probability threshold for plausibility
        y_target: Target labels for counterfactuals (optional)

    Returns:
        dict: Dictionary containing computed metrics
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


def _get_log_prob_threshold(
    gen_model: torch.nn.Module,
    dataset: DictConfig,
    batch_size: int,
    log_prob_quantile: float,
) -> float:
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")
    return log_prob_threshold


def _create_gen_model_from_cfg(
    model_cfg: DictConfig,
    dataset: DictConfig,
    model_path: str,
    dequantizer: GroupDequantizer,
) -> torch.nn.Module:
    model_target = model_cfg.model._target_
    is_ceflow_gmm = "CeFlowGMM" in model_target
    init_kwargs = {"features": dataset.X_train.shape[1]}
    if is_ceflow_gmm:
        init_kwargs["categorical_groups"] = dataset.categorical_features_lists
        init_kwargs["n_classes"] = len(np.unique(dataset.y_train))
    else:
        context_features = model_cfg.get("context_features", 1)
        init_kwargs["context_features"] = context_features
    gen_model = instantiate(model_cfg.model, **init_kwargs)
    if model_cfg.train_model:
        train_loader = dataset.train_dataloader(
            batch_size=model_cfg.batch_size,
            shuffle=True,
            noise_lvl=model_cfg.noise_lvl,
        )
        test_loader = dataset.test_dataloader(
            batch_size=model_cfg.batch_size,
            shuffle=False,
        )
        fit_kwargs = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "epochs": model_cfg.epochs,
            "patience": model_cfg.patience,
            "lr": model_cfg.lr,
            "checkpoint_path": model_path,
        }
        if "dequantizer" in inspect.signature(gen_model.fit).parameters:
            fit_kwargs["dequantizer"] = dequantizer
        gen_model.fit(**fit_kwargs)
        gen_model.save(model_path)
    else:
        gen_model.load(model_path)
    gen_model.eval()
    return gen_model


@hydra.main(config_path="./conf", config_name="ceflow_config", version_base="1.2")
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    file_dataset = instantiate(cfg.dataset)
    dataset = MethodDataset(file_dataset, preprocessing_pipeline)
    dequantizer = GroupDequantizer(dataset.categorical_features_lists)

    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, density_model_path, save_folder = set_model_paths(
            cfg, fold=fold_n
        )
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        dequantizer.fit(dataset.X_train)

        output_folder = os.path.dirname(disc_model_path)
        flow_model_name = cfg.flow_model.model._target_.split(".")[-1]
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        if cfg.experiment.relabel_with_disc_model:
            flow_model_path = os.path.join(
                output_folder,
                f"flow_model_{flow_model_name}_relabeled_by_{disc_model_name}.pt",
            )
        else:
            flow_model_path = os.path.join(
                output_folder, f"flow_model_{flow_model_name}.pt"
            )

        flow_model = _create_gen_model_from_cfg(
            cfg.flow_model, dataset, flow_model_path, dequantizer
        )
        density_model = _create_gen_model_from_cfg(
            cfg.gen_model, dataset, density_model_path, dequantizer
        )

        dataset.X_train = dequantizer.transform(dataset.X_train)
        log_prob_threshold = _get_log_prob_threshold(
            density_model,
            dataset,
            cfg.counterfactuals_params.batch_size,
            cfg.counterfactuals_params.log_prob_quantile,
        )
        dataset.X_train = dequantizer.inverse_transform(dataset.X_train)

        Xs_cfs, Xs, ys_orig, ys_target, model_returned, cf_search_time = (
            search_counterfactuals(
                cfg, dataset, flow_model, disc_model, save_folder, dequantizer
            )
        )

        density_model = DequantizationWrapper(density_model, dequantizer)
        metrics = calculate_metrics(
            gen_model=density_model,
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
        df_metrics["cf_search_time"] = cf_search_time
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
