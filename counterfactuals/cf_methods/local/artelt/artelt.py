import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.discriminative_models.base import BaseDiscModel
from counterfactuals.cf_methods.artelt.artelth20.plausible_counterfactuals import (
    HighDensityEllipsoids,
    PlausibleCounterfactualOfHyperplaneClassifier,
)


class Artelt(BaseCounterfactual):
    def __init__(self, disc_model: BaseDiscModel, **kwargs) -> None:
        self.disc_model = disc_model
        self.density_estimators = {}
        self.kernel_density_estimators = {}
        self.ellipsoids = {}
        self.cf = {}

    def fit_density_estimators(self, X_train, y_train):
        labels = np.unique(y_train)
        for label in labels:
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
            print(f"bandwidth: {bandwidth}")

            cv = GridSearchCV(
                estimator=GaussianMixture(covariance_type="full"),
                param_grid={"n_components": range(2, 10)},
                n_jobs=-1,
                cv=5,
            )
            cv.fit(X_)
            n_components = cv.best_params_["n_components"]
            print(f"n_components: {n_components}")
            # Build density estimators
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(X_)

            de = GaussianMixture(
                n_components=n_components, covariance_type="full", random_state=42
            )
            de.fit(X_)

            self.density_estimators[label] = de
            self.kernel_density_estimators[label] = kde

            # Compute densities and ellipsoids
            densities_training_samples, densities_training_samples_ex = (
                self._compute_densities(X_, de)
            )
            cluster_prob_ = de.predict_proba(X_)
            density_threshold = np.median(densities_training_samples)
            print(f"density_threshold: {density_threshold}")

            self.ellipsoids[label] = HighDensityEllipsoids(
                X_,
                densities_training_samples_ex,
                cluster_prob_,
                de.means_,
                de.covariances_,
                density_threshold,
            ).compute_ellipsoids()

            # Prepare counterfactual generator
            disc_model_coef_ = (
                list(self.disc_model.parameters())[0].detach().cpu().numpy()
            )
            disc_model_intercept_ = (
                list(self.disc_model.parameters())[1].detach().cpu().numpy()
            )

            self.cf[label] = PlausibleCounterfactualOfHyperplaneClassifier(
                disc_model_coef_,
                disc_model_intercept_,
                n_dims=X_train.shape[1],
                ellipsoids_r=self.ellipsoids[label],
                gmm_weights=de.weights_,
                gmm_means=de.means_,
                gmm_covariances=de.covariances_,
                density_threshold=density_threshold,
            )
            print(f"Plausible counterfactual generator for label {label} fitted")

    def _compute_densities(self, X_, de):
        densities_training_samples = []
        densities_training_samples_ex = []
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

            densities_training_samples.append(np.min(z))
            densities_training_samples_ex.append(z)

        return np.array(densities_training_samples), np.array(
            densities_training_samples_ex
        )

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        self.fit_density_estimators(X_train, y_train)

        x_orig = X.reshape(1, -1)
        y_pred = self.disc_model.predict(x_orig).item()
        y_target = np.abs(1 - y_pred)

        xcf = self.cf[y_target].compute_counterfactual(x_orig.squeeze(), y=y_target)

        if xcf is None:
            explanation = np.empty_like(x_orig)
            explanation[:] = np.nan
        else:
            explanation = xcf

        return ExplanationResult(
            x_cfs=explanation.reshape(1, -1),
            y_cf_targets=np.array([y_target]),
            x_origs=x_orig,
            y_origs=np.array([y_origin]),
        )

    def explain_dataloader(
        self, dataloader: DataLoader, *args, **kwargs
    ) -> ExplanationResult:
        if not self.density_estimators:
            raise ValueError("Density estimators not fitted")

        Xs, ys = dataloader.dataset.tensors

        Xs_cfs = []
        ys_target = []
        model_returned = []

        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            x_orig = X.numpy().reshape(1, -1)
            y_pred = self.disc_model.predict(x_orig).item()
            y_target = np.abs(1 - y_pred).astype(int)
            # print(f"y_target: {y_target}")
            # print(f"x_orig: {x_orig}")

            xcf = self.cf[y_target].compute_counterfactual(x_orig.squeeze(), y=y_target)

            if xcf is None:
                print("No counterfactual found!")
                explanation = np.empty_like(x_orig)
                explanation[:] = np.nan
                model_returned.append(False)
            else:
                explanation = xcf
                model_returned.append(True)

            Xs_cfs.append(explanation)
            ys_target.append(y_target)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        ys_target = np.array(ys_target)

        return Xs_cfs, Xs, ys, ys_target, model_returned

        # return ExplanationResult(
        #     x_cfs=Xs_cfs, y_cf_targets=ys_target, x_origs=Xs.numpy(), y_origs=ys.numpy()
        # )
