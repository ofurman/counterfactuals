import logging
import os
from time import time

import hydra
import neptune
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from counterfactuals.cf_methods.artelth20.plausible_counterfactuals import (
    HighDensityEllipsoids,
    PlausibleCounterfactualOfHyperplaneClassifier,
)
from counterfactuals.generative_models import BaseGenModel
from counterfactuals.metrics.metrics import evaluate_cf

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf/other_methods", config_name="config_artelt", version_base="1.2"
)
def main(cfg: DictConfig):
    logger.info("Initializing Neptune run")
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
    # run["parameters/counterfactuals"] = cfg.counterfactuals
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = dataset_name
    run["parameters/reference_method"] = "Artelt"
    # run["parameters/pca_dim"] = cfg.pca_dim
    run.wait()

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)

    disc_model_path = os.path.join(output_folder, f"disc_model_{disc_model_name}.pt")
    if cfg.experiment.relabel_with_disc_model:
        gen_model_path = os.path.join(
            output_folder,
            f"gen_model_{gen_model_name}_relabeled_by_{disc_model_name}.pt",
        )
    else:
        gen_model_path = os.path.join(output_folder, f"gen_model_{gen_model_name}.pt")

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
    # if cfg.pca_dim is not None and cfg.pca_dim < dataset.X_train.shape[1]:
    #     raise NotImplementedError("PCA is not supported yet")
    # pca = PCA(n_components=cfg.pca_dim)
    # pca.fit(X_train)

    # projection_matrix = pca.components_  # Projection matrix
    # projection_mean_sub = pca.mean_

    # X_train = np.dot(X_train - projection_mean_sub, projection_matrix.T)
    # X_test = np.dot(X_test - projection_mean_sub, projection_matrix.T)

    # Start ArteltH20 Method
    # For each class, fit density estimators
    logger.info("Fitting density estimators")
    density_estimators = {}
    kernel_density_estimators = {}
    densities_training_samples = {}
    densities_training_samples_ex = {}
    ellipsoids = {}
    cf = {}
    labels = np.unique(y_train)
    for label in labels:
        # Get all samples with the 'correct' label
        idx = y_train == label
        X_ = X_train[idx, :]

        # Optimize hyperparameters
        cv = GridSearchCV(
            estimator=KernelDensity(),
            param_grid={"bandwidth": np.arange(0.1, 10.0, 0.05)},
            n_jobs=-1,
            cv=5,
        )
        cv.fit(X_)
        bandwidth = cv.best_params_["bandwidth"]
        print("bandwidth: {0}".format(bandwidth))

        cv = GridSearchCV(
            estimator=GaussianMixture(covariance_type="full"),
            param_grid={"n_components": range(2, 10)},
            n_jobs=-1,
            cv=5,
        )
        cv.fit(X_)
        n_components = cv.best_params_["n_components"]
        print("n_components: {0}".format(n_components))

        # Build density estimators
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(X_)

        de = GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=42
        )
        de.fit(X_)

        density_estimators[label] = de
        kernel_density_estimators[label] = kde

        # Compute media NLL of training samples
        # TODO: Move this to the outer loop
        # from scipy.stats import multivariate_normal
        X_ = X_train[y_train == label, :]
        densities_training_samples[label] = []
        densities_training_samples_ex[label] = []
        for j in range(X_.shape[0]):
            x = X_[j, :]
            z = []
            dim = x.shape[0]
            for i in range(de.weights_.shape[0]):
                x_i = de.means_[i]
                w_i = de.weights_[i]
                cov = de.covariances_[i]
                cov = np.linalg.inv(cov)

                b = (
                    -2.0 * np.log(w_i)
                    + dim * np.log(2.0 * np.pi)
                    - np.log(np.linalg.det(cov))
                )
                z.append(np.dot(x - x_i, np.dot(cov, x - x_i)) + b)  # NLL

            densities_training_samples[label].append(np.min(z))
            densities_training_samples_ex[label].append(z)

        densities_training_samples[label] = np.array(densities_training_samples[label])
        densities_training_samples_ex[label] = np.array(
            densities_training_samples_ex[label]
        )

        cluster_prob_ = de.predict_proba(X_)
        density_threshold = np.median(densities_training_samples[label])
        # Compute high density ellipsoids - constraint: test if sample is included in ellipsoid -> this is the same as the proposed constraint but nummerically much more stable, in particular when we add a dimensionality reduction from a high dimensional space to a low dimensional space
        ellipsoids[label] = HighDensityEllipsoids(
            X_,
            densities_training_samples_ex[label],
            cluster_prob_,
            de.means_,
            de.covariances_,
            density_threshold,
        ).compute_ellipsoids()

        # Compute counterfactul with proposed density constraint
        disc_model_coef_ = list(disc_model.parameters())[0].detach().cpu().numpy()
        disc_model_intercept_ = list(disc_model.parameters())[1].detach().cpu().numpy()

        cf[label] = PlausibleCounterfactualOfHyperplaneClassifier(
            disc_model_coef_,
            disc_model_intercept_,
            n_dims=X_train.shape[1],
            ellipsoids_r=ellipsoids[label],
            gmm_weights=de.weights_,
            gmm_means=de.means_,
            gmm_covariances=de.covariances_,
            # projection_matrix=projection_matrix,
            # projection_mean_sub=projection_mean_sub,
            density_threshold=density_threshold,
        )

    # For each point in the test set
    # Compute and plot counterfactual without density constraints
    logger.info("n_test_samples: %d", X_test.shape[0])
    Xs_cfs = []
    model_returned = []
    Xs_cfs_times = []
    time_start = time()
    for i in tqdm(range(X_test.shape[0])):
        # x_orig = X_test[i,:]
        x_orig_orig = X_test[i, :]
        y_orig = y_test[i]
        y_target = np.abs(1 - y_orig)

        # if disc_model.predict(np.array([x_orig_orig])) == y_target:  # Model already predicts target label!
        #     print("Requested prediction already satisfied")
        #     Xs_cfs.append(x_orig_orig)
        #     continue

        # # Compute and plot counterfactual WITH kernel density constraints
        # idx = y_train == y_target
        # X_ = X_train[idx, :]

        # # Build density estimator
        # de = density_estimators[y_target]
        # kde = kernel_density_estimators[y_target]

        xcf_t2 = time()
        xcf2 = cf[y_target].compute_counterfactual(x_orig_orig, y=y_target)
        xcf_t2 = time() - xcf_t2
        Xs_cfs_times.append(xcf_t2)
        if xcf2 is None:
            print("No counterfactual found!")
            model_returned.append(False)
        else:
            Xs_cfs.append(xcf2)
            model_returned.append(True)

    cf_search_time = time() - time_start
    run["metrics/eval_time"] = np.mean(cf_search_time)
    run["metrics/avg_time_one_cf"] = np.mean(Xs_cfs_times)

    Xs_cfs = np.array(Xs_cfs, dtype=np.float32).squeeze()
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    # Xs_cfs = pca.inverse_transform(Xs_cfs)
    train_dataloader_for_log_prob = dataset.train_dataloader(
        batch_size=cfg.counterfactuals.batch_size, shuffle=False
    )
    delta = torch.median(gen_model.predict_log_prob(train_dataloader_for_log_prob))
    run["parameters/delta"] = delta
    print(delta)
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

    log_df = pd.DataFrame(metrics, index=[0])

    log_df.to_csv(os.path.join(output_folder, "metrics_artelt.csv"), index=False)

    run.stop()


if __name__ == "__main__":
    main()
