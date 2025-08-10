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
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.cf_methods.lice.lice import LiCE
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
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create counterfactuals using LiCE method
    """
    cf_method_name = "LiCE"
    disc_model.eval()
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    logger.info("Filtering out target class data for counterfactual generation")
    target_class = 1
    ys_pred = disc_model.predict(X_test)
    Xs = dataset.X_test[ys_pred != target_class]
    ys_orig = ys_pred[ys_pred != target_class]

    logger.info("Creating counterfactual model")
    # Convert data to pandas DataFrame for LiCE
    X_train_df = pd.DataFrame(X_train, columns=dataset.features[:-1])
    y_train_df = pd.DataFrame(y_train, columns=[dataset.features[-1]])
    X_test_df = pd.DataFrame(X_test, columns=dataset.features[:-1])
    y_test_df = pd.DataFrame(y_test, columns=[dataset.features[-1]])

    # Create data handler and SPN
    from counterfactuals.cf_methods.lice.data.DataHandler import DataHandler
    from counterfactuals.cf_methods.lice.SPN import SPN

    dhandler = DataHandler(X_train_df, y_train_df)
    spn = SPN(
        np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
        dhandler,
        normalize_data=True,
    )

    # Create and save ONNX model
    import torch.onnx

    os.makedirs(f"{save_folder}/models", exist_ok=True)
    dummy_input = torch.randn(1, X_train.shape[1])
    torch.onnx.export(
        disc_model,
        dummy_input,
        f"{save_folder}/models/nn.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    lice = LiCE(
        spn,
        nn_path=f"{save_folder}/models/nn.onnx",
        data_handler=dhandler,
    )

    logger.info("Calculating log_prob_threshold")
    train_data = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
    lls = spn.compute_ll(train_data)
    log_prob_threshold = np.median(lls)
    run["parameters/log_prob_threshold"] = log_prob_threshold
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")

    logger.info("Handling counterfactual generation")
    time_start = time()
    Xs_cfs = []
    model_returned = []
    ys_target = []
    for i, sample in enumerate(Xs):
        try:
            sample_ll = spn.compute_ll(np.concatenate([sample, y_test[i : i + 1]]))[0]
            enc_sample = dhandler.encode(
                pd.DataFrame([sample], columns=dataset.features[:-1])
            )
            prediction = disc_model.predict(enc_sample) > 0

            # Generate counterfactual
            time_limit = 600  # Default time limit in seconds
            if hasattr(cfg, "counterfactuals_params") and hasattr(
                cfg.counterfactuals_params, "time_limit"
            ):
                time_limit = cfg.counterfactuals_params.time_limit

            cf = lice.generate_counterfactual(
                sample,
                not prediction,
                ll_threshold=log_prob_threshold,
                n_counterfactuals=1,
                time_limit=time_limit,
                leaf_encoding="histogram",
                spn_variant="lower",
                solver_name="cbc",
            )

            print(cf)
            if len(cf) > 0:
                Xs_cfs.append(cf[0])
                model_returned.append(True)
                ys_target.append(1 - prediction)
            else:
                Xs_cfs.append(sample)
                model_returned.append(False)
                ys_target.append(1 - prediction)
        except Exception as e:
            logger.error(f"Error generating counterfactual for sample {i}: {str(e)}")
            Xs_cfs.append(sample)
            model_returned.append(False)
            ys_target.append(1 - int(ys_orig[i]))

    Xs_cfs = np.array(Xs_cfs)
    model_returned = np.array(model_returned)
    ys_target = np.array(ys_target)
    cf_search_time = time() - time_start
    run["metrics/cf_search_time"] = cf_search_time

    counterfactuals_path = os.path.join(
        save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
    )
    pd.DataFrame(Xs_cfs, columns=dataset.features[:-1]).to_csv(
        counterfactuals_path, index=False
    )
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


@hydra.main(config_path="./conf", config_name="globe_ce_config", version_base="1.2")
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
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

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
            y_train=dataset.y_train.reshape(-1),
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
