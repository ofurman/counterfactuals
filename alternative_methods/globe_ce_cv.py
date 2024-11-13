import logging
import os
from time import time

import hydra
import numpy as np
import neptune
from neptune.utils import stringify_unsupported
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import classification_report

from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.metrics.metrics import (
    local_outlier_factor,
    isolation_forest_metric,
)
from counterfactuals.cf_methods.ares import AReS, dnn_normalisers, lr_normalisers
from counterfactuals.cf_methods.globe_ce import GLOBE_CE
from counterfactuals.metrics.metrics import (
    continuous_distance,
    categorical_distance,
    distance_l2_jaccard,
    distance_mad_hamming,
    sparsity,
    perc_valid_cf,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)


NORMALISERS = {
    "dnn": dnn_normalisers,
    "lr": lr_normalisers,
}


@hydra.main(config_path="../conf", config_name="config_globe_ce", version_base="1.2")
def main(cfg: DictConfig):
    logger.info("Initializing Neptune run")
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    disc_model_name = cfg.disc_model
    gen_model_name = cfg.gen_model.model._target_.split(".")[-1]
    dataset_name = cfg.dataset._target_.split(".")[-1]
    output_folder = os.path.join(cfg.experiment.output_folder, dataset_name)
    save_folder = os.path.join(output_folder, "globe_ce")
    os.makedirs(save_folder, exist_ok=True)
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]

    os.makedirs(output_folder, exist_ok=True)
    logger.info("Creatied output folder %s", output_folder)

    logger.info("Logging parameters")

    log_df = pd.DataFrame()

    logger.info("Loading dataset")
    cf_dataset = instantiate(cfg.dataset, method="ares")

    for fold_n, (_, _, _, _) in enumerate(cf_dataset.get_cv_splits(n_splits=5)):
        disc_model_path = os.path.join(
            output_folder, f"disc_model_{fold_n}_{disc_model_name}.pt"
        )
        gen_model_path = os.path.join(
            output_folder, f"gen_model_{fold_n}_{gen_model_name}.pt"
        )

        logger.info("Training discriminator model")
        binary_datasets = [
            "MoonsDataset",
            "LawDataset",
            "HelocDataset",
            "AuditDataset",
            "BlobsDataset",  # temoporary
            "DigitsDataset",  # temoporary
        ]
        num_classes = (
            1 if dataset_name in binary_datasets else len(np.unique(cf_dataset.y_train))
        )
        disc_model = instantiate(
            cfg.disc_model.model,
            input_size=cf_dataset.X_train.shape[1],
            target_size=num_classes,
        )
        train_dataloader = cf_dataset.train_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=True, noise_lvl=0
        )
        test_dataloader = cf_dataset.test_dataloader(
            batch_size=cfg.disc_model.batch_size, shuffle=False
        )

        # disc_model.fit(
        #     train_dataloader,
        #     test_dataloader,
        #     epochs=cfg.disc_model.epochs,
        #     lr=cfg.disc_model.lr,
        #     patience=cfg.disc_model.patience,
        #     checkpoint_path=disc_model_path,
        # )
        # disc_model.save(disc_model_path)

        logger.info("Evaluating discriminator model")
        disc_model.load(disc_model_path)

        print(
            classification_report(
                cf_dataset.y_test, disc_model.predict(cf_dataset.X_test)
            )
        )
        report = classification_report(
            cf_dataset.y_test, disc_model.predict(cf_dataset.X_test), output_dict=True
        )
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(
                output_folder, f"eval_disc_model_{fold_n}_{disc_model_name}.csv"
            )
        )

        run[f"{fold_n}/disc_model"].upload(disc_model_path)

        if cfg.experiment.relabel_with_disc_model:
            cf_dataset.y_train = disc_model.predict(cf_dataset.X_train).detach().numpy()
            cf_dataset.y_test = disc_model.predict(cf_dataset.X_test).detach().numpy()

        logger.info("Training generative model")
        gen_model = instantiate(
            cfg.gen_model.model,
            features=cf_dataset.X_train.shape[1],
            context_features=1,
        )
        # gen_model.load(gen_model_path)
        gen_model.fit(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            num_epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
            learning_rate=cfg.gen_model.lr,
            checkpoint_path=gen_model_path,
            # neptune_run=run,
        )
        gen_model.save(gen_model_path)
        run[f"{fold_n}/gen_model"].upload(gen_model_path)

        train_ll = gen_model.predict_log_prob(train_dataloader).mean().item()
        test_ll = gen_model.predict_log_prob(test_dataloader).mean().item()
        pd.DataFrame({"train_ll": [train_ll], "test_ll": [test_ll]}).to_csv(
            os.path.join(save_folder, f"eval_gen_model_{fold_n}_{gen_model_name}.csv")
        )

        normalisers = NORMALISERS.get(cfg.model, {dataset_name: False})

        cf_dataset.X_train = pd.DataFrame(
            cf_dataset.X_train, columns=cf_dataset.feature_columns
        )
        cf_dataset.X_test = pd.DataFrame(
            cf_dataset.X_test, columns=cf_dataset.feature_columns
        )

        X = pd.DataFrame(cf_dataset.X_test, columns=cf_dataset.feature_columns).astype(
            np.float32
        )

        cf_dataset.X_train.columns = [str(col) for col in cf_dataset.X_train.columns]
        cf_dataset.X_test.columns = [str(col) for col in cf_dataset.X_test.columns]
        X.columns = [str(col) for col in X.columns]

        ares = AReS(
            model=disc_model,
            dataset=cf_dataset,
            X=X,
            dropped_features=[],
            n_bins=10,
            ordinal_features=[],
            normalise=normalisers[dataset_name],
            constraints=[20, 7, 10],
            dataset_name=dataset_name,
        )
        bin_widths = ares.bin_widths

        ordinal_features = (
            ["Present-Employment"] if dataset_name == "german_credit" else []
        )
        globe_ce = GLOBE_CE(
            model=disc_model,
            dataset=cf_dataset,
            X=X,
            affected_subgroup=None,
            dropped_features=[],
            ordinal_features=ordinal_features,
            delta_init="zeros",
            normalise=None,
            bin_widths=bin_widths,
            monotonicity=None,
            p=1,
            dataset_name=dataset_name,
        )
        cf_dataset.X_train = cf_dataset.X_train.to_numpy()
        train_dataloader_for_log_prob = cf_dataset.train_dataloader(
            batch_size=cfg.counterfactuals.batch_size, shuffle=False
        )
        median_log_prob = torch.median(
            gen_model.predict_log_prob(train_dataloader_for_log_prob)
        )
        run[f"{fold_n}/parameters/median_log_prob"] = median_log_prob

        logger.info("Handling counterfactual generation")
        time_start = time()

        best_delta = get_best_delta(globe_ce)
        best_k_s = get_best_k_s(globe_ce, best_delta)
        Xs_cfs = get_counterfactuals(globe_ce, best_delta, best_k_s)

        cf_search_time = np.mean(time() - time_start)
        run[f"{fold_n}/metrics/cf_search_time"] = cf_search_time
        counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
        pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
        run["counterfactuals"].upload(counterfactuals_path)

        model_returned = np.ones(Xs_cfs.shape[0]).astype(bool)

        logger.info("Calculating metrics")

        X_aff = globe_ce.x_aff
        metrics = evaluate_globe_ce(
            X_cf=Xs_cfs,
            X_aff=X_aff,
            X_train=cf_dataset.X_train,
            X_test=torch.tensor(cf_dataset.X_test.to_numpy()),
            y_test=cf_dataset.y_test,
            disc_model=disc_model,
            gen_model=gen_model,
            model_returned=model_returned,
            median_log_prob=median_log_prob
        )

        print(metrics)
        run[f"{fold_n}/metrics/cf"] = stringify_unsupported(metrics)
        metrics["time"] = cf_search_time
        log_df = pd.concat([log_df, pd.DataFrame(metrics, index=[fold_n])])

    logger.info("Finalizing and stopping run")
    log_df.to_csv(
        os.path.join(save_folder, f"metrics_{disc_model_name}_cv.csv"), index=False
    )
    run["metrics"].upload(
        os.path.join(output_folder, f"metrics_{disc_model_name}_cv.csv")
    )
    run.stop()


def get_best_k_s(globe_ce, best_delta):
    _, cos_s, k_s = globe_ce.scale(
        best_delta, disable_tqdm=False, vector=True
    )  # Algorithm 1, Line 3
    _, min_costs_idxs = globe_ce.min_scalar_costs(
        cos_s, return_idxs=True, inf=True
    )  # Implicitly computes Algorithm 1, Lines 4-6, returning minimum costs per input and their indices in the k vector
    best_k_s = k_s[min_costs_idxs.astype(np.int16)]
    return best_k_s


def get_counterfactuals(globe_ce, best_delta, best_k_s):
    muls_ = best_k_s.reshape(-1, 1) @ best_delta.reshape(1, -1)
    Xs_cfs = globe_ce.x_aff + muls_
    return Xs_cfs


def get_best_delta(globe_ce):
    globe_ce.sample(
        n_sample=1000,
        magnitude=2,
        sparsity_power=1,
        idxs=None,
        n_features=2,
        disable_tqdm=False,  # 2 random features chosen at each sample, no sparsity smoothing (p=1)
        plot=False,
        seed=0,
        scheme="random",
        dropped_features=[],
    )  # plot=False
    delta = globe_ce.best_delta  # pick best delta
    return delta


def evaluate_globe_ce(
    X_cf,
    X_aff,
    X_train,
    X_test,
    y_test,
    disc_model,
    gen_model,
    model_returned,
    median_log_prob,
):
    categorical_features = []
    continuous_features = list(range(X_cf.shape[1]))

    model_returned_smth = np.sum(model_returned) / len(model_returned)

    lof_scores_xs, lof_scores_cfs = local_outlier_factor(X_train, X_aff, X_cf)
    isolation_forest_scores_xs, isolation_forest_scores_cfs = isolation_forest_metric(
        X_train, X_aff, X_cf
    )

    ys_cfs_disc_pred = torch.tensor(disc_model.predict(X_cf))

    valid_cf_disc_metric = perc_valid_cf(
        torch.zeros_like(ys_cfs_disc_pred), y_cf=ys_cfs_disc_pred
    )

    ys_pred = disc_model.predict(X_aff)
    y_target = torch.abs(1 - ys_pred)

    X_cf = torch.tensor(X_cf, dtype=torch.float32)
    gen_log_probs_xs = gen_model(X_aff, torch.zeros(X_aff.shape[0]))
    gen_log_probs_cf = gen_model(X_cf, torch.tensor(y_target).type(torch.float32))
    flow_prob_condition_acc = torch.sum(median_log_prob < gen_log_probs_cf) / len(
        gen_log_probs_cf
    )

    hamming_distance_metric = categorical_distance(
        X=X_aff,
        X_cf=X_cf,
        categorical_features=categorical_features,
        metric="hamming",
        agg="mean",
    )
    jaccard_distance_metric = categorical_distance(
        X=X_aff,
        X_cf=X_cf,
        categorical_features=categorical_features,
        metric="jaccard",
        agg="mean",
    )
    manhattan_distance_metric = continuous_distance(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="cityblock",
        X_all=X_test,
    )
    euclidean_distance_metric = continuous_distance(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="euclidean",
        X_all=X_test,
    )
    mad_distance_metric = continuous_distance(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="mad",
        X_all=X_test,
    )
    l2_jaccard_distance_metric = distance_l2_jaccard(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
    mad_hamming_distance_metric = distance_mad_hamming(
        X=X_aff,
        X_cf=X_cf,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        X_all=X_test,
        agg="mean",
    )

    X_aff, X_cf = torch.tensor(X_aff), torch.tensor(X_cf)
    sparsity_metric = sparsity(X_aff, X_cf)

    metrics = {
        "model_returned_smth": model_returned_smth,
        "valid_cf_disc": valid_cf_disc_metric,
        "dissimilarity_proximity_categorical_hamming": hamming_distance_metric,
        "dissimilarity_proximity_categorical_jaccard": jaccard_distance_metric,
        "dissimilarity_proximity_continuous_manhatan": manhattan_distance_metric,
        "dissimilarity_proximity_continuous_euclidean": euclidean_distance_metric,
        "dissimilarity_proximity_continuous_mad": mad_distance_metric,
        "distance_l2_jaccard": l2_jaccard_distance_metric,
        "distance_mad_hamming": mad_hamming_distance_metric,
        "sparsity": sparsity_metric,
    }

    metrics.update(
        {
            "flow_log_density_cfs": gen_log_probs_cf.mean().item(),
            "flow_log_density_xs": gen_log_probs_xs.mean().item(),
            "flow_prob_condition_acc": flow_prob_condition_acc.item(),
            "lof_scores_xs": lof_scores_xs.mean(),
            "lof_scores_cfs": lof_scores_cfs.mean(),
            "isolation_forest_scores_xs": isolation_forest_scores_xs.mean(),
            "isolation_forest_scores_cfs": isolation_forest_scores_cfs.mean(),
        }
    )
    return metrics


if __name__ == "__main__":
    main()
