import logging
import os
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from counterfactuals.cf_methods.sace.blackbox import BlackBox
from counterfactuals.cf_methods.sace.casebased_sace import CaseBasedSACE
from counterfactuals.generative_models.base import BaseGenModel
from counterfactuals.metrics.metrics import evaluate_cf

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def generate_cf(dataset, disc_model):
    X_train, X_test = dataset.X_train, dataset.X_test

    time_start = time()
    # Start CBCE Method
    variable_features = dataset.numerical_features + dataset.categorical_features
    metric = ("euclidean", "jaccard")
    #         ('cosine', 'jaccard'),
    #         # ('euclidean', 'hamming')
    cf = CaseBasedSACE(
        variable_features=variable_features,
        weights=None,
        metric=metric,
        feature_names=None,
        continuous_features=dataset.numerical_features,
        categorical_features_lists=dataset.categorical_features_lists,
        normalize=False,
        random_samples=None,
        diff_features=10,
        tolerance=0.001,
    )
    bb = BlackBox(disc_model)
    cf.fit(bb, X_train)

    Xs_cfs = []
    model_returned = []
    for x in tqdm(X_test):
        x_cf = cf.get_counterfactuals(x, k=1)
        Xs_cfs.append(x_cf)
        model_returned.append(True)

    cf_search_time = time() - time_start
    # run["metrics/avg_time_one_cf"] = cf_search_time / X_test.shape[0]
    # run["metrics/eval_time"] = np.mean(cf_search_time)

    Xs_cfs = np.array(Xs_cfs).squeeze()
    return model_returned, Xs_cfs, cf_search_time


@hydra.main(
    config_path="../conf/other_methods", config_name="config_cbce", version_base="1.2"
)
def main(cfg: DictConfig):
    logger.info("Initializing Neptune run")
    # run = neptune.init_run(
    #     mode="async" if cfg.neptune.enable else "offline",
    #     project=cfg.neptune.project,
    #     api_token=cfg.neptune.api_token,
    #     tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    # )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    dataset_name = cfg.dataset._target_.split(".")[-1]
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    save_folder = os.path.join(output_folder, cfg.reference_method)
    os.makedirs(save_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    # Log parameters using Hydra config
    logger.info("Logging parameters")
    # run["parameters/dataset"] = cfg.dataset._target_.split(".")[-1]
    # run["parameters/disc_model/model_name"] = disc_model_name
    # run["parameters/disc_model"] = cfg.disc_model
    # run["parameters/gen_model/model_name"] = gen_model_name
    # run["parameters/gen_model"] = cfg.gen_model
    # # run["parameters/counterfactuals"] = cfg.counterfactuals
    # run["parameters/experiment"] = cfg.experiment
    # run["parameters/dataset"] = dataset_name
    # run["parameters/reference_method"] = "Artelt"
    # # run["parameters/pca_dim"] = cfg.pca_dim
    # run.wait()

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
        num_classes = 1 if disc_model_name == "LogisticRegression" else len(np.unique(dataset.y_train))
        disc_model = instantiate(
            cfg.disc_model.model,
            input_size=dataset.X_train.shape[1],
            target_size=num_classes,
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


        model_returned, Xs_cfs, cf_search_time = generate_cf(dataset, disc_model)

        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{disc_model_name}_{fold_n}.csv"
        )
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)

        # Xs_cfs = pd.read_csv(counterfactuals_path).values.astype(np.float32)
        # model_returned = ~np.isnan(Xs_cfs[:, 0])
        # cf_search_time = pd.read_csv(
        #     os.path.join(save_folder, f"metrics_{disc_model_name}_cv.csv")
        # )["time"].iloc[fold_n]
        # run["counterfactuals"].upload(counterfactuals_path)

        # Xs_cfs = pca.inverse_transform(Xs_cfs)
        train_dataloader_for_log_prob = dataset.train_dataloader(
            batch_size=cfg.counterfactuals.batch_size, shuffle=False
        )
        delta = torch.median(gen_model.predict_log_prob(train_dataloader_for_log_prob))
        # run["parameters/delta"] = delta
        print(delta)
        metrics = evaluate_cf(
            disc_model=disc_model,
            gen_model=gen_model,
            X_cf=Xs_cfs,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features,
            continuous_features=dataset.numerical_features,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=dataset.X_test,
            y_test=dataset.y_test.reshape(-1),
            delta=delta,
        )
        # run["metrics/cf"] = metrics

        metrics["time"] = cf_search_time

        log_df = pd.concat([log_df, pd.DataFrame(metrics, index=[fold_n])])

    log_df.to_csv(
        os.path.join(save_folder, f"metrics_{disc_model_name}_cv.csv"), index=False
    )

    # run.stop()


if __name__ == "__main__":
    main()
