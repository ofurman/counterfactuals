import logging
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from counterfactuals.cf_methods.global_methods.globe_ce import GLOBE_CE
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
        return data.copy(), columns

    group_lookup = {column: base for base, group_cols in groups.items() for column in group_cols}
    added_groups = set()
    for column in columns:
        base = group_lookup.get(column)
        if base is None:
            dataset.features_tree[column] = []
            continue
        if base in added_groups:
            continue
        grouped_columns = [feature for feature in columns if group_lookup.get(feature) == base]
        dataset.features_tree[base] = grouped_columns
        added_groups.add(base)

    return data.copy(), columns


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
    return data_oh, features


def compute_bin_widths(dataset, data, n_bins=10):
    bin_widths = {}
    for feature in data.columns:
        if feature in dataset.categorical_features:
            continue

        try:
            categories = pd.cut(data[feature].astype(float), bins=n_bins).cat.categories
        except ValueError as err:
            logger.warning("Skipping bin width computation for feature %s: %s", feature, err)
            continue

        if len(categories) == 0:
            logger.warning(
                "Skipping bin width computation for feature %s: no categories returned",
                feature,
            )
            continue

        bin_widths[feature] = float(categories.length[-1])

    return bin_widths


class GLOBECEPipelineRunner(PipelineRunner):
    """Pipeline runner for GLOBE-CE counterfactual generation."""

    cf_method_name = "GLOBE_CE"

    def load_dataset(self):
        file_dataset = instantiate(self.cfg.dataset)
        return MethodDataset(file_dataset, self.preprocessing_pipeline)

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model.eval()
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        target_class = self.cfg.counterfactuals_params.target_class

        minmax_scaler = dataset.preprocessing_pipeline.get_step("minmax")

        X_test_unscaled = minmax_scaler._inverse_transform_array(dataset.X_test)
        data_oh, features = one_hot(
            dataset, pd.DataFrame(X_test_unscaled, columns=dataset.features)
        )

        def predict_fn(x):
            x_array = x.values if isinstance(x, pd.DataFrame) else x
            x_scaled = minmax_scaler._transform_array(x_array)
            return disc_model.predict(x_scaled)

        logger.info("Filtering out target class data for counterfactual generation")
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

        logger.info("Handling counterfactual generation")
        time_start = time()
        ys_target = np.full_like(ys_orig, target_class)
        explanation_result = cf_method.explain(
            y_origin=ys_orig,
            y_target=ys_target,
        )
        Xs_cfs = explanation_result.x_cfs
        Xs_cfs = minmax_scaler._transform_array(Xs_cfs)
        Xs_cfs, model_returned = align_counterfactuals_with_factuals(Xs_cfs, Xs)
        cf_search_time = np.mean(time() - time_start)
        logger.info(f"Counterfactual search completed in {cf_search_time:.4f} seconds")

        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=Xs,
            y_orig=ys_orig,
            y_target=ys_target,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )


@hydra.main(config_path="./conf", config_name="globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    # GLOBE-CE uses torch_dtype first, then minmax (different order)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("torch_dtype", TorchDataTypeStep()),
            ("minmax", MinMaxScalingStep()),
        ]
    )
    runner = GLOBECEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
