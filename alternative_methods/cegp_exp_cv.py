import hydra
import os
import torch
import logging
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import neptune
from hydra.utils import instantiate
from tqdm import tqdm

from omegaconf import DictConfig

from alibi.explainers import CounterFactualProto

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.generative_models.base import BaseGenModel


logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf/other_methods", config_name="config_cegp", version_base="1.2"
)
def main(cfg: DictConfig):
    tf.compat.v1.disable_eager_execution()

    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    run["parameters/disc_model/model_name"] = disc_model_name
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model/model_name"] = gen_model_name
    run["parameters/gen_model"] = cfg.gen_model
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = dataset_name
    run["parameters/reference_method"] = "CEGP"
    run.wait()

    log_df = pd.DataFrame()

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    for fold_n, (_, _, _, _) in enumerate(dataset.get_cv_splits(n_splits=5)):
        disc_model_path = os.path.join(
            output_folder, f"disc_model_{fold_n}_{disc_model_name}.pt"
        )
        if cfg.experiment.relabel_with_disc_model:
            gen_model_path = os.path.join(
                output_folder,
                f"gen_model_{fold_n}_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
            )
        else:
            gen_model_path = os.path.join(
                output_folder, f"gen_model_{fold_n}_{gen_model_name}.pt"
            )

        logger.info("Loading discriminator model")
        disc_model = instantiate(
            cfg.disc_model.model,
            input_size=dataset.X_train.shape[1],
            target_size=len(np.unique(dataset.y_train)),
        )
        disc_model.load(disc_model_path)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train)
            dataset.y_test = disc_model.predict(dataset.X_test)

        logger.info("Loading generator model")
        gen_model: BaseGenModel = instantiate(
            cfg.gen_model.model, features=dataset.X_train.shape[1], context_features=1
        )
        gen_model.load(gen_model_path)

        X_train, X_test, y_train, y_test = (
            dataset.X_train,
            dataset.X_test,
            dataset.y_train.reshape(-1),
            dataset.y_test.reshape(-1),
        )

        shape = (1,) + X_train.shape[1:]  # instance shape
        beta = 0.01
        c_init = 1.0
        c_steps = 5
        max_iterations = 500
        # rng = (-1., 1.)  # scale features between -1 and 1
        # rng_shape = (1,) + X_train.shape[1:]
        feature_range = (
            X_train.min(axis=0).reshape(
                shape
            ),  # feature range for the perturbed instance
            X_train.max(axis=0).reshape(shape),
        )

        cf = CounterFactualProto(
            disc_model.predict_proba,
            shape,
            beta=beta,
            # cat_vars=cat_vars_ohe,
            # ohe=True,  # OHE flag
            max_iterations=max_iterations,
            feature_range=feature_range,
            c_init=c_init,
            c_steps=c_steps,
        )

        cf.fit(X_train.astype(np.float32), d_type="abdm", disc_perc=[25, 50, 75])

        Xs_cfs = []
        model_returned = []
        start_time = time()
        for X, y in tqdm(zip(X_test, y_test), total=len(X_test)):
            try:
                explanation = cf.explain(X.reshape(1, -1)).cf
            except Exception as e:
                explanation = None
                logger.info("Error in CounterfactualProto: %s", e)
            if explanation is None:
                model_returned.append(False)
            else:
                Xs_cfs.append(explanation["X"])
                model_returned.append(True)

        cf_search_time = time() - start_time
        run["metrics/avg_time_one_cf"] = (cf_search_time) / X_test.shape[0]
        run["metrics/eval_time"] = np.mean(cf_search_time)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        run["counterfactuals"].upload(counterfactuals_path)

        train_dataloader_for_log_prob = dataset.train_dataloader(
            batch_size=cfg.counterfactuals.batch_size, shuffle=False
        )
        delta = torch.median(gen_model.predict_log_prob(train_dataloader_for_log_prob))
        run["parameters/delta"] = delta
        metrics = evaluate_cf(
            disc_model=disc_model,
            gen_model=gen_model,
            X_cf=Xs_cfs,
            y_target=y_test,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            delta=delta,
        )
        run["metrics/cf"] = metrics

        metrics["time"] = cf_search_time
        log_df = pd.concat([log_df, pd.DataFrame(metrics, index=[fold_n])])
    log_df.to_csv(os.path.join(output_folder, "metrics_cegp_cv.csv"), index=False)

    run.stop()


if __name__ == "__main__":
    main()
