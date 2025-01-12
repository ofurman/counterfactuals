import logging
import os
import hydra
import numpy as np
import pandas as pd
from time import time
import torch
import neptune
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.utils
from sklearn.preprocessing import LabelEncoder

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.cf_methods.ares import AReS
from counterfactuals.pipelines.nodes.helper_nodes import log_parameters, set_model_paths
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def one_hot(dataset, data):
    """
    Improvised method for one-hot encoding the data

    Input: data (whole dataset)
    Outputs: data_oh (one-hot encoded data)
                features (list of feature values after one-hot encoding)
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
            data_encode[x] = pd.cut(
                data_encode[x].apply(lambda x: float(x)), bins=dataset.n_bins
            )
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


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    run: neptune.Run,
    save_folder: str,
) -> torch.nn.Module:
    """
    Create counterfactuals using CEM method
    """
    cf_method_name = "ARES"
    disc_model.eval()
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    X_test_unscaled = dataset.feature_transformer.inverse_transform(dataset.X_test)
    data_oh, features = one_hot(
        dataset, pd.DataFrame(X_test_unscaled, columns=dataset.features[:-1])
    )

    def predict_fn(x):
        x_scaled = dataset.feature_transformer.transform(x)
        return disc_model.predict(x_scaled).detach().numpy().flatten()

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = 1
    ys_pred = predict_fn(X_test_unscaled)
    Xs = dataset.X_test[ys_pred != target_class]
    ys_orig = ys_pred[ys_pred != target_class]

    logger.info("Creating counterfactual model")
    cf_method = AReS(
        predict_fn=predict_fn,
        dataset=dataset,
        X=pd.DataFrame(X_test_unscaled, columns=dataset.features[:-1]),
        dropped_features=[],
        n_bins=10,
        ordinal_features=[],
        normalise=False,
        constraints=[20, 7, 10],
    )
    logger.info("Calculating log_prob_threshold")
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals_params.batch_size, shuffle=False
    )
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader_for_log_prob),
        cfg.counterfactuals_params.log_prob_quantile,
    )
    run["parameters/log_prob_threshold"] = log_prob_threshold
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")

    logger.info("Handling counterfactual generation")
    time_start = time()
    Xs_cfs = cf_method.explain()
    Xs_cfs = dataset.feature_transformer.transform(Xs_cfs)
    ys_orig = dataset.y_transformer.transform(ys_orig.reshape(-1, 1))
    ys_target = np.zeros_like(ys_orig)
    ys_target[:, 1] = 1
    model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)
    cf_search_time = np.mean(time() - time_start)
    run["metrics/cf_search_time"] = cf_search_time

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    return Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target, model_returned


def calculate_metrics(
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    Xs_cfs: np.ndarray,
    model_returned: np.ndarray,
    categorical_features: list,
    continuous_features: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    median_log_prob: float,
    run: neptune.Run,
    y_target: np.ndarray = None,
):
    """
    Calculate metrics for counterfactuals
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
    run["metrics/cf"] = stringify_unsupported(metrics)
    logger.info(f"Metrics:\n{stringify_unsupported(metrics)}")
    return metrics


@hydra.main(config_path="./conf", config_name="ares_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    log_parameters(cfg, run)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset, shuffle=False)

    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder, run)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = dataset.y_transformer.transform(
                disc_model.predict(dataset.X_train).detach().numpy().reshape(-1, 1)
            )
            dataset.y_test = dataset.y_transformer.transform(
                disc_model.predict(dataset.X_test).detach().numpy().reshape(-1, 1)
            )

        gen_model = create_gen_model(cfg, dataset, gen_model_path, run)

        Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target, model_returned = (
            search_counterfactuals(
                cfg, dataset, gen_model, disc_model, run, save_folder
            )
        )

        metrics = calculate_metrics(
            gen_model=gen_model,
            disc_model=disc_model,
            Xs_cfs=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
            run=run,
        )

        run[f"metrics/cf/fold_{fold_n}"] = stringify_unsupported(metrics)
        df_metrics = pd.DataFrame(metrics, index=[0])
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False
        )

    run.stop()


if __name__ == "__main__":
    main()
