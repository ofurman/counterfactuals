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
from sklearn.metrics import classification_report
import torch.utils

from counterfactuals.metrics.metrics import evaluate_cf


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

torch.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def log_parameters(cfg: DictConfig, run: neptune.Run):
    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]

    if cfg.get("disc_model"):
        run["parameters/disc_model/model_name"] = cfg.disc_model.model._target_.split(
            "."
        )[-1]
        run["parameters/disc_model"] = stringify_unsupported(cfg.disc_model)

    if cfg.get("gen_model"):
        run["parameters/gen_model/model_name"] = cfg.gen_model.model._target_.split(
            "."
        )[-1]
        run["parameters/gen_model"] = stringify_unsupported(cfg.gen_model)

    run["parameters/counterfactuals"] = cfg.counterfactuals_params
    run["parameters/counterfactuals/method_name"] = (
        cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
    )
    run.wait()


def set_model_paths(cfg: DictConfig):
    """
    Saves results in the output folder with the following structure:
    output_folder/dataset_name/_disc_model_name.pt
    output_folder/dataset_name/_gen_model_name.pt
    output_folder/dataset_name/method_name/results
    """
    # Set paths for saving models
    logger.info("Setting model paths")
    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    cf_method_name = cfg.counterfactuals_params.cf_method._target_.split(".")[-1]

    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)
    save_folder = os.path.join(output_folder, cf_method_name)
    os.makedirs(save_folder, exist_ok=True)
    logger.info("Created save folder %s", save_folder)

    disc_model_path = os.path.join(output_folder, f"disc_model_{disc_model_name}.pt")
    if cfg.experiment.relabel_with_disc_model:
        gen_model_path = os.path.join(
            output_folder,
            f"gen_model_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
        )
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{gen_model_name}.pt")

    logger.info("Disc model path: %s", disc_model_path)
    logger.info("Gen model path: %s", gen_model_path)

    return disc_model_path, gen_model_path, save_folder


def create_disc_model(
    cfg: DictConfig,
    dataset: DictConfig,
    disc_model_path: str,
    save_folder: str,
    run: neptune.Run,
) -> torch.nn.Module:
    """
    Create and train a discriminator model
    """

    logger.info("Creating discriminator model")
    binary_datasets = [
        "MoonsDataset",
        "LawDataset",
        "HelocDataset",
        "AuditDataset",
    ]
    dataset_name = cfg.dataset._target_.split(".")[-1]
    num_classes = (
        1 if dataset_name in binary_datasets else len(np.unique(dataset.y_train))
    )
    disc_model = instantiate(
        cfg.disc_model.model,
        input_size=dataset.X_train.shape[1],
        target_size=num_classes,
    )
    if cfg.disc_model.train_model:
        logger.info("Training discriminator model")
        train_dataloader = dataset.train_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=True, noise_lvl=0
        )
        test_dataloader = dataset.test_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=False
        )
        disc_model.fit(
            train_dataloader,
            test_dataloader,
            epochs=cfg.disc_model.epochs,
            lr=cfg.disc_model.lr,
            patience=cfg.disc_model.patience,
            checkpoint_path=disc_model_path,
        )
        disc_model.save(disc_model_path)
    else:
        logger.info("Loading discriminator model")
        disc_model.load(disc_model_path)

    disc_model.eval()
    logger.info("Evaluating discriminator model")
    print(classification_report(dataset.y_test, disc_model.predict(dataset.X_test)))
    report = classification_report(
        dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True
    )
    # pd.DataFrame(report).transpose().to_csv(
    #     os.path.join(save_folder, f"eval_disc_model_{disc_model_name}.csv")
    # )
    run["metrics/disc_model"] = stringify_unsupported(report)
    logger.info(
        f"Discriminator model evaluation results:\n {stringify_unsupported(report)}"
    )
    return disc_model


def create_gen_model(
    cfg: DictConfig, dataset: DictConfig, gen_model_path: str, run: neptune.Run
) -> torch.nn.Module:
    """
    Create and train a generative model
    """
    train_dataloader = dataset.train_dataloader(
        batch_size=cfg.gen_model.batch_size,
        shuffle=True,
        noise_lvl=cfg.gen_model.noise_lvl,
    )
    test_dataloader = dataset.test_dataloader(
        batch_size=cfg.gen_model.batch_size, shuffle=False
    )
    logger.info("Creating generative model")
    gen_model = instantiate(
        cfg.gen_model.model, features=dataset.X_train.shape[1], context_features=1
    )
    if cfg.gen_model.train_model:
        logger.info("Training generative model")
        gen_model.fit(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            num_epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
            learning_rate=cfg.gen_model.lr,
            checkpoint_path=gen_model_path,
            neptune_run=run,
        )
        gen_model.save(gen_model_path)
    else:
        logger.info("Loading generative model")
        gen_model.load(gen_model_path)

    gen_model.eval()
    logger.info("Evaluating generative model")
    train_ll = gen_model.predict_log_prob(train_dataloader).mean().item()
    test_ll = gen_model.predict_log_prob(test_dataloader).mean().item()
    # pd.DataFrame({"train_ll": [train_ll], "test_ll": [test_ll]}).to_csv(
    #     os.path.join(save_folder, f"eval_gen_model_{gen_model_name}.csv")
    # )
    run["metrics/gen_model"] = {"train_ll": train_ll, "test_ll": test_ll}
    logger.info(
        f"Generative model evaluation results:\n train_ll: {train_ll:.4f}, test_ll: {test_ll:.4f}"
    )
    return gen_model


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

    logger.info("Creating counterfactual model")
    cf = instantiate(
        cfg.counterfactuals_params.cf_method,
        N=dataset.X_test.shape[0],
        D=dataset.X_test.shape[1],
        gen_model=gen_model,
        disc_model=disc_model,
        disc_model_criterion=instantiate(cfg.counterfactuals_params.disc_loss),
        neptune_run=run,
    )

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = cfg.counterfactuals_params.target_class
    X_test_origin = dataset.X_test[dataset.y_test != target_class]
    y_test_origin = dataset.y_test[dataset.y_test != target_class]
    X_test_target = dataset.X_test[dataset.y_test == target_class]

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
    deltas, Xs, ys_orig, ys_target, _ = cf.explain_dataloader(
        dataloader=cf_dataloader,
        epochs=cfg.counterfactuals_params.epochs,
        lr=cfg.counterfactuals_params.lr,
        patience=cfg.counterfactuals_params.patience,
        alpha=cfg.counterfactuals_params.alpha,
        alpha_s=cfg.counterfactuals_params.alpha_s,
        alpha_k=cfg.counterfactuals_params.alpha_k,
        beta=cfg.counterfactuals_params.beta,
        log_prob_threshold=log_prob_threshold,
    )
    cf_search_time = np.mean(time() - time_start)
    run["metrics/cf_search_time"] = cf_search_time
    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_no_plaus_{cf_method_name}_{disc_model_name}.csv"
    )
    M, S, D = deltas[0].get_matrices()

    Xs_cfs = Xs + deltas[0]().detach().numpy()
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)
    return Xs_cfs, log_prob_threshold, S, X_test_target


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
    S_matrix: np.ndarray,
    X_test_target: np.ndarray,
    run: neptune.Run,
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
        S_matrix=S_matrix,
        X_test_target=X_test_target,
    )
    run["metrics/cf"] = stringify_unsupported(metrics)
    logger.info(f"Metrics:\n{stringify_unsupported(metrics)}")
    return metrics


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
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

    Xs_cfs, log_prob_threshold, S, X_test_target = search_counterfactuals(
        cfg, dataset, gen_model, disc_model, run, save_folder
    )

    metrics = calculate_metrics(
        gen_model=gen_model,
        disc_model=disc_model,
        Xs_cfs=Xs_cfs,
        model_returned=np.ones(Xs_cfs.shape[0]).astype(bool),
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=dataset.X_train,
        y_train=dataset.y_train.reshape(-1),
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        median_log_prob=log_prob_threshold,
        S_matrix=S.detach().numpy(),
        X_test_target=X_test_target,
        run=run,
    )
    print(metrics)
    run.stop()


if __name__ == "__main__":
    main()
