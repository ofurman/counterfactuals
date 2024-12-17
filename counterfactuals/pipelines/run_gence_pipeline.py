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
from torch.utils.data import DataLoader


from counterfactuals.metrics import CFMetrics
from counterfactuals.generative_models import MaskedAutoregressiveFlow
from counterfactuals.cf_methods.gence.preprocess import PairDistanceDataset
from counterfactuals.pipelines.nodes.helper_nodes import log_parameters, set_model_paths
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def prepare_data_for_gence(dataset):
    class_zero = dataset.X_train[dataset.y_train == 0]
    class_one = dataset.X_train[dataset.y_train == 1]
    pair_dataset_train = PairDistanceDataset(class_zero, class_one, length=5000)

    def collate_fn(batch):
        X, y = zip(*batch)
        X = torch.stack(X)
        y = torch.stack(y)
        noise = torch.randn_like(X) * 0.03
        noise = torch.randn_like(y) * 0.03
        X = X + noise
        y = y + noise
        return X, y

    train_dataloader = DataLoader(
        pair_dataset_train, batch_size=256, shuffle=True, collate_fn=collate_fn
    )

    class_zero = dataset.X_test[dataset.y_test == 0]
    class_one = dataset.X_test[dataset.y_test == 1]

    pair_dataset_test = PairDistanceDataset(class_zero, class_one)

    test_dataloader = DataLoader(pair_dataset_test, batch_size=2048, shuffle=False)
    return train_dataloader, test_dataloader


def search_counterfactuals(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    run: neptune.Run,
    save_folder: str,
) -> torch.nn.Module:
    """
    Create a counterfactual model
    """

    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test
    y_test_origin = dataset.y_test
    y_test_target = np.abs(dataset.y_test - 1)
    # X_test_target = dataset.X_test[dataset.y_test == target_class]

    cf_train_dataloader, cf_test_dataloader = prepare_data_for_gence(dataset)

    logger.info("Creating counterfactual model")
    cf_method = MaskedAutoregressiveFlow(
        features=dataset.X_test.shape[1],
        hidden_features=16,
        num_blocks_per_layer=4,
        num_layers=8,
        context_features=dataset.X_test.shape[1],
    )
    cf_method.fit(
        cf_train_dataloader,
        cf_test_dataloader,
        num_epochs=1000,
        learning_rate=1e-3,
        patience=70,
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
    Xs_cfs = []
    with torch.no_grad():
        for x in dataset.X_test:
            points, log_prob = cf_method.sample_and_log_prob(
                1, context=torch.from_numpy(np.array([x]))
            )
            Xs_cfs.append(points)
    Xs_cfs = torch.stack(Xs_cfs).squeeze().numpy()
    cf_search_time = np.mean(time() - time_start)

    run["metrics/cf_search_time"] = cf_search_time
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)
    return Xs_cfs, X_test_origin, log_prob_threshold, y_test_origin, y_test_target


def calculate_metrics(
    X_cfs: np.ndarray,
    gen_model: torch.nn.Module,
    disc_model: torch.nn.Module,
    dataset: DictConfig,
    prob_threshold: float,
    run: neptune.Run,
):
    """
    Calculate metrics for counterfactuals
    """
    logger.info("Calculating metrics")
    metrics = CFMetrics(
        X_cf=X_cfs,
        y_target=np.abs(dataset.y_test - 1),
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        gen_model=gen_model,
        disc_model=disc_model,
        continuous_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
        prob_plausibility_threshold=prob_threshold,
    )
    run["metrics/cf"] = stringify_unsupported(metrics)
    logger.info(f"Metrics:\n{stringify_unsupported(metrics)}")
    return metrics.calc_all_metrics()


@hydra.main(config_path="./conf", config_name="gence_config", version_base="1.2")
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
    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder, run)

    if cfg.experiment.relabel_with_disc_model:
        dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
        dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

    gen_model = create_gen_model(cfg, dataset, gen_model_path, run)

    # Custom code
    Xs_cfs, Xs, log_prob_threshold, ys_orig, ys_target = search_counterfactuals(
        cfg, dataset, gen_model, disc_model, run, save_folder
    )

    metrics = calculate_metrics(
        X_cfs=Xs_cfs,
        gen_model=gen_model,
        disc_model=disc_model,
        dataset=dataset,
        prob_threshold=log_prob_threshold,
        run=run,
    )
    print(metrics)
    run.stop()


if __name__ == "__main__":
    main()
