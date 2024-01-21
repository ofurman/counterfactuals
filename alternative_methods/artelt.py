import logging
import os
from time import time
from uuid import uuid4

import hydra
import mlflow
import neptune
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra.utils import instantiate
from joblib import dump, load
from omegaconf import DictConfig
from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

from counterfactuals.artelt.plausible_counterfactuals import (
    HighDensityEllipsoids, PlausibleCounterfactualOfDecisionTree,
    PlausibleCounterfactualOfHyperplaneClassifier)
from counterfactuals.metrics.metrics import (
    valid_cf,
    categorical_distance,
    continuous_distance,
    distance_l2_jaccard,
    distance_mad_hamming, kde_density,
    perc_valid_actionable_cf,
    perc_valid_cf, plausibility,
    sparsity,
    evaluate_cf
)
from counterfactuals.utils import process_classification_report

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf/other_methods", config_name="config_artelt", version_base="1.2")
def main(cfg: DictConfig):
    run = neptune.init_run(
        mode="async" if cfg.neptune.enable else "offline",
        project=cfg.neptune.project,
        api_token=cfg.neptune.api_token,
        tags=list(cfg.neptune.tags) if "tags" in cfg.neptune else None,
    )

    output_folder = cfg.experiment.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Log parameters using Hydra config
    run["parameters/experiment"] = cfg.experiment
    run["parameters/dataset"] = cfg.dataset
    run["parameters/disc_model"] = cfg.disc_model
    run["parameters/gen_model"] = "Artelt"

    available_disc_models = ["LogisticRegression", "DecisionTreeClassifier"]
    if cfg.disc_model._target_.split(".")[-1] not in available_disc_models:
        raise ValueError(f"Disc model not supported. Please choose one of {available_disc_models}")
    use_decision_tree = cfg.disc_model._target_.split(".")[-1] == "DecisionTreeClassifier"

    dataset = instantiate(cfg.dataset)
    X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test

    disc_model = instantiate(cfg.disc_model)
    disc_model.fit(dataset.X_train, dataset.y_train.reshape(-1))
    report = classification_report(dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True)

    mlflow.log_metrics(process_classification_report(report, prefix="disc_test"))
    run["metrics"] = process_classification_report(report, prefix="disc_test")

    disc_model_path = os.path.join(output_folder, f"disc_model_{uuid4()}.joblib")
    dump(disc_model, disc_model_path)
    run["disc_model"].upload(disc_model_path)

    X_test_pred_path = os.path.join(output_folder, "X_test_pred.csv")
    pd.DataFrame(disc_model.predict(dataset.X_test)).to_csv(X_test_pred_path, index=False)
    run["X_test_pred"].upload(X_test_pred_path)

    y_train = disc_model.predict(X_train)
    y_test = disc_model.predict(X_test)

    # ###############################################################################################################################################
    # ###############################################################################################################################################
    # ###############################################################################################################################################

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
            estimator=KernelDensity(), param_grid={"bandwidth": np.arange(0.1, 10.0, 0.05)}, n_jobs=-1, cv=5
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

        de = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
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

                b = -2.0 * np.log(w_i) + dim * np.log(2.0 * np.pi) - np.log(np.linalg.det(cov))
                z.append(np.dot(x - x_i, np.dot(cov, x - x_i)) + b)  # NLL

            densities_training_samples[label].append(np.min(z))
            densities_training_samples_ex[label].append(z)

        densities_training_samples[label] = np.array(densities_training_samples[label])
        densities_training_samples_ex[label] = np.array(densities_training_samples_ex[label])

        cluster_prob_ = de.predict_proba(X_)
        density_threshold = np.median(densities_training_samples[label])
        # Compute high density ellipsoids - constraint: test if sample is included in ellipsoid -> this is the same as the proposed constraint but nummerically much more stable, in particular when we add a dimensionality reduction from a high dimensional space to a low dimensional space
        ellipsoids[label] = HighDensityEllipsoids(
            X_, densities_training_samples_ex[label], cluster_prob_, de.means_, de.covariances_, density_threshold
        ).compute_ellipsoids()

        # Compute counterfactul with proposed density constraint
        
        if use_decision_tree is False:
            cf[label] = PlausibleCounterfactualOfHyperplaneClassifier(
                disc_model.coef_,
                disc_model.intercept_,
                n_dims=X_train.shape[1],
                ellipsoids_r=ellipsoids[label],
                gmm_weights=de.weights_,
                gmm_means=de.means_,
                gmm_covariances=de.covariances_,
                # projection_matrix=projection_matrix,
                # projection_mean_sub=projection_mean_sub,
                density_threshold=density_threshold,
            )
        else:
            cf[label] = PlausibleCounterfactualOfDecisionTree(
                disc_model,
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
    Xs_cfs_times = []
    for i in tqdm(range(X_test.shape[0])):
        # x_orig = X_test[i,:]
        x_orig_orig = X_test[i, :]
        y_orig = y_test[i]
        y_target = np.abs(1 - y_orig)

        if disc_model.predict([x_orig_orig]) == y_target:  # Model already predicts target label!
            print("Requested prediction already satisfied")
            Xs_cfs.append(x_orig_orig)
            continue

        # Compute and plot counterfactual WITH kernel density constraints
        idx = y_train == y_target
        X_ = X_train[idx, :]

        # Build density estimator
        de = density_estimators[y_target]
        kde = kernel_density_estimators[y_target]

        xcf_t2 = time()
        xcf2 = cf[y_target].compute_counterfactual(x_orig_orig, y=y_target)
        xcf_t2 = time() - xcf_t2
        Xs_cfs_times.append(xcf_t2)
        if xcf2 is None:
            print("No counterfactual found!")
            Xs_cfs.append(x_orig_orig)
            continue
        Xs_cfs.append(xcf2)

    run["metrics/avg_time_one_cf"] = np.mean(Xs_cfs_times)

    Xs_cfs = np.array(Xs_cfs).squeeze()
    counterfactuals_path = os.path.join(output_folder, "counterfactuals.csv")
    pd.DataFrame(Xs_cfs).to_csv(counterfactuals_path, index=False)
    run["counterfactuals"].upload(counterfactuals_path)

    metrics = evaluate_cf(
        disc_model=disc_model,
        X=X_test,
        X_cf=Xs_cfs,
        categorical_features=dataset.categorical_features,
        continuous_features=dataset.numerical_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    run["metrics/cf"] = metrics

    run.stop()


if __name__ == "__main__":
    main()
