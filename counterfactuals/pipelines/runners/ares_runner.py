from __future__ import annotations

import copy
import logging
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from counterfactuals.cf_methods import AReS
from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult
from counterfactuals.pipelines.utils import align_counterfactuals_with_factuals
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def _set_dataset_attribute(dataset, attribute, value):
    try:
        setattr(dataset, attribute, value)
        return
    except AttributeError:
        pass

    if hasattr(dataset, "file_dataset"):
        setattr(dataset.file_dataset, attribute, value)
        return

    raise


def _infer_one_hot_category(base_feature, column):
    if not column.startswith(base_feature):
        return column

    suffix = column[len(base_feature) :]
    for sep in (" = ", "__", "=", "_"):
        if suffix.startswith(sep):
            return suffix[len(sep) :]
    return suffix.lstrip(" _=")


def _build_features_tree_from_one_hot(dataset, data):
    groups = getattr(dataset, "one_hot_feature_groups", None)
    if groups is None and hasattr(dataset, "file_dataset"):
        groups = getattr(dataset.file_dataset, "one_hot_feature_groups", None)

    dataset.bins = {}
    dataset.bins_tree = {}
    dataset.features_tree = {}
    dataset.n_bins = None

    columns = list(data.columns)
    if not groups:
        dataset.features_tree = {col: [] for col in columns}
        return data.copy(), columns

    group_lookup = {
        column: base_feature
        for base_feature, group_columns in groups.items()
        for column in group_columns
    }

    data_transformed = data.copy()
    transformed_columns = []
    for column in columns:
        base_feature = group_lookup.get(column)
        if base_feature is None:
            dataset.features_tree[column] = []
            transformed_columns.append(column)
            continue

        category = _infer_one_hot_category(base_feature, column)
        feature_value = f"{base_feature} = {category}" if category else column
        dataset.features_tree.setdefault(base_feature, []).append(feature_value)
        transformed_columns.append(feature_value)

    data_transformed.columns = transformed_columns
    _set_dataset_attribute(dataset, "features", transformed_columns)
    _set_dataset_attribute(
        dataset,
        "categorical_features",
        [feature for feature, values in dataset.features_tree.items() if values],
    )
    return data_transformed, transformed_columns


def one_hot(dataset, data):
    if getattr(dataset, "one_hot_feature_groups", None) or (
        hasattr(dataset, "file_dataset")
        and getattr(dataset.file_dataset, "one_hot_feature_groups", None)
    ):
        return _build_features_tree_from_one_hot(dataset, data)

    label_encoder = LabelEncoder()
    data_encode = data.copy()
    dataset.bins = {}
    dataset.bins_tree = {}
    dataset.features_tree = {}
    dataset.n_bins = None

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
    _set_dataset_attribute(dataset, "features", features)
    return data_oh, features


def _feature_columns(dataset):
    columns = list(dataset.features)
    target = None
    if hasattr(dataset, "target"):
        target = dataset.target
    elif hasattr(dataset, "config"):
        target = getattr(dataset.config, "target", None)
    if target is not None and target in columns:
        columns = [col for col in columns if col != target]
    return columns


def _get_feature_transformer(dataset):
    transformer = getattr(dataset, "feature_transformer", None)
    if transformer is not None:
        return transformer
    if hasattr(dataset, "preprocessing_pipeline"):
        return dataset.preprocessing_pipeline.get_step("minmax")
    return None


def _ensure_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().numpy().flatten()
    return np.asarray(array).flatten()


class AReSPipelineRunner(PipelineRunner):
    """Pipeline runner for AReS counterfactual generation."""

    cf_method_name = "ARES"

    def load_dataset(self):
        dataset = instantiate(self.cfg.dataset)
        if hasattr(dataset, "train_dataloader"):
            return dataset

        preprocessing_pipeline = PreprocessingPipeline(
            [
                ("minmax", MinMaxScalingStep()),
                ("torch_dtype", TorchDataTypeStep()),
            ]
        )
        return MethodDataset(dataset, preprocessing_pipeline)

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model.eval()
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]

        feature_transformer = _get_feature_transformer(dataset)
        minmax_scaler = None
        if feature_transformer is None:
            minmax_scaler = dataset.preprocessing_pipeline.get_step("minmax")
            X_test_unscaled = minmax_scaler._inverse_transform_array(dataset.X_test)
        else:
            if hasattr(feature_transformer, "_inverse_transform_array"):
                X_test_unscaled = feature_transformer._inverse_transform_array(dataset.X_test)
            else:
                X_test_unscaled = feature_transformer.inverse_transform(dataset.X_test)
        feature_columns = _feature_columns(dataset)
        ares_dataset = copy.deepcopy(dataset)
        X_test_for_ares, _ = one_hot(
            ares_dataset, pd.DataFrame(X_test_unscaled, columns=feature_columns)
        )

        def predict_fn_raw(x):
            x_array = x.values if isinstance(x, pd.DataFrame) else x
            if feature_transformer is not None:
                if hasattr(feature_transformer, "_transform_array"):
                    x_scaled = feature_transformer._transform_array(x_array)
                else:
                    x_scaled = feature_transformer.transform(x_array)
            else:
                x_scaled = minmax_scaler._transform_array(x_array)
            preds = disc_model.predict(x_scaled)
            return _ensure_numpy(preds)

        logger.info("Filtering out target class data for counterfactual generation")
        target_class = getattr(self.cfg.counterfactuals_params, "target_class", 1)
        ys_pred = predict_fn_raw(X_test_unscaled)
        mask = ys_pred != target_class
        Xs_for_ares = X_test_for_ares.loc[mask].reset_index(drop=True)
        Xs = dataset.X_test[mask]
        ys_orig = ys_pred[mask]

        predict_fn_for_cf = (
            (lambda x: 1 - predict_fn_raw(x)) if target_class == 0 else predict_fn_raw
        )

        logger.info("Creating counterfactual model")
        apriori_threshold = float(
            getattr(self.cfg.counterfactuals_params, "apriori_threshold", 0.6)
        )
        n_bins = int(getattr(self.cfg.counterfactuals_params, "n_bins", 10))
        max_triples_eval = int(getattr(self.cfg.counterfactuals_params, "max_triples_eval", 5000))
        cf_method = AReS(
            predict_fn=predict_fn_for_cf,
            dataset=ares_dataset,
            X=Xs_for_ares,
            dropped_features=[],
            n_bins=n_bins,
            ordinal_features=[],
            normalise=False,
            constraints=[20, 7, 10],
        )

        logger.info("Handling counterfactual generation")
        time_start = time()
        ys_target = np.full_like(ys_orig, target_class)
        explanation_result = cf_method.explain(
            apriori_threshold=apriori_threshold,
            max_triples_eval=max_triples_eval,
            y_origin=ys_orig,
            y_target=ys_target,
        )
        Xs_cfs = explanation_result.x_cfs
        if Xs_cfs.shape[0] > 0:
            if feature_transformer is not None:
                if hasattr(feature_transformer, "_transform_array"):
                    Xs_cfs = feature_transformer._transform_array(Xs_cfs)
                else:
                    Xs_cfs = feature_transformer.transform(Xs_cfs)
            else:
                Xs_cfs = minmax_scaler._transform_array(Xs_cfs)
        Xs_cfs, model_returned = align_counterfactuals_with_factuals(Xs_cfs, Xs)
        cf_search_time = np.mean(time() - time_start)
        logger.info(f"Counterfactual search time: {cf_search_time:.2f} seconds")

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="ares_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = AReSPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
