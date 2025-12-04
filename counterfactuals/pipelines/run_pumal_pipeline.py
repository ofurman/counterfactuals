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
from collections import defaultdict

from counterfactuals.metrics.metrics import evaluate_cf_for_pumal
from counterfactuals.cf_methods.pumal import PUMAL
from counterfactuals.pipelines.nodes.helper_nodes import log_parameters, set_model_paths
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


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
    origin_class = cfg.counterfactuals_params.origin_class
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[np.argmax(dataset.y_test, axis=1) == origin_class]
    y_test_origin = dataset.y_test[np.argmax(dataset.y_test, axis=1) == origin_class]
    # X_test_target = dataset.X_test[dataset.y_test == target_class]

    logger.info("Creating counterfactual model")
    disc_model_criterion = instantiate(cfg.counterfactuals_params.disc_model_criterion)

    cf_method: PUMAL = PUMAL(
        cf_method_type=cfg.counterfactuals_params.cf_method.cf_method_type,
        K=cfg.counterfactuals_params.cf_method.K,
        X=X_test_origin,
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=disc_model_criterion,
        not_actionable_features=dataset.not_actionable_features,
        neptune_run=None,
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
    cf_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test_origin).float(),
            torch.tensor(y_test_origin).float(),
        ),
        batch_size=cfg.counterfactuals_params.batch_size,
        shuffle=False,
    )
    time_start = time()
    delta, Xs, ys_orig, ys_target = cf_method.explain_dataloader(
        dataloader=cf_dataloader,
        target_class=target_class,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        patience=cfg.counterfactuals_params.patience,
        alpha_dist=cfg.counterfactuals_params.alpha_dist,
        alpha_plaus=cfg.counterfactuals_params.alpha_plaus,
        alpha_class=cfg.counterfactuals_params.alpha_class,
        alpha_s=cfg.counterfactuals_params.alpha_s,
        alpha_k=cfg.counterfactuals_params.alpha_k,
        alpha_d=cfg.counterfactuals_params.alpha_d,
        log_prob_threshold=log_prob_threshold,
        decrease_loss_patience=cfg.counterfactuals_params.decrease_loss_patience,
    )

    cf_search_time = np.mean(time() - time_start)
    run["metrics/cf_search_time"] = cf_search_time
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    M, S, D = delta.get_matrices()

    Xs_cfs = Xs + delta().detach().numpy()
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    return Xs_cfs, Xs, log_prob_threshold, M, S, D, ys_orig, ys_target, cf_search_time


@hydra.main(config_path="./conf", config_name="pumal_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logger.info("Initializing Neptune run")
    run = defaultdict(list)

    log_parameters(cfg, run)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

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

        Xs_cfs, Xs, log_prob_threshold, M, S, D, ys_orig, ys_target, cf_search_time = (
            search_counterfactuals(
                cfg, dataset, gen_model, disc_model, run, save_folder
            )
        )

        logger.info("Calculating metrics")
        metrics = evaluate_cf_for_pumal(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=Xs_cfs,
            model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=Xs,
            y_test=ys_orig,
            y_target=ys_target,
            median_log_prob=log_prob_threshold,
            S_matrix=S.detach().numpy(),
            D_matrix=D.detach().numpy(),
            X_test_target=Xs,
        )
        logger.info(f"Metrics:\n{stringify_unsupported(metrics)}")
        df_metrics = pd.DataFrame(metrics, index=[0])
        disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
        df_metrics["time"] = cf_search_time
        df_metrics.to_csv(
            os.path.join(save_folder, f"cf_metrics_{disc_model_name}_"
                                      f"p_{cfg.counterfactuals_params.alpha_plaus}_"
                                      f"k_{cfg.counterfactuals_params.alpha_k}_"
                                      f"s_{cfg.counterfactuals_params.alpha_s}.csv"), index=False
        )


if __name__ == "__main__":
    main()
