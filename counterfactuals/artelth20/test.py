# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
import random
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_boston, load_wine
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from plausible_counterfactuals import HighDensityEllipsoids, PlausibleCounterfactualOfHyperplaneClassifier, PlausibleCounterfactualOfDecisionTree


def load_house_prices(file_path="housepricesdataset.npz"):
    X, y = load_boston(return_X_y=True)
    y = y >= 20
    y = y.astype(np.int).flatten()

    return X, y


if __name__ == "__main__":
    use_decision_tree = False   # If False, softmax regression is used!

    # Load data set
    X, y = load_iris(return_X_y=True);pca_dim=None
    #X, y = load_breast_cancer(return_X_y=True);pca_dim=5
    #X, y = load_house_prices();pca_dim=10
    #X, y = load_wine(return_X_y=True);pca_dim=8
    X, y = load_digits(return_X_y=True);pca_dim=40

    X, y = shuffle(X, y, random_state=42)

    # k-fold cross validation
    scores_with_density_constraint = []
    scores_without_density_constraint = []

    original_data = []
    original_data_labels = []
    cfs_with_density_constraint = []
    cfs_without_density_constraint = []
    cfs_target_label = []
    computation_time_without_density_constraint = []
    computation_time_with_density_constraint = []
    distances_with_density_constraint = []
    distances_without_density_constraint = []

    kf = KFold(n_splits=5, random_state=42)
    for train_index, test_index in kf.split(X):
        # Split data into training and test set
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Choose target labels
        y_test_target = []
        labels = np.unique(y)
        for i in range(X_test.shape[0]):
            y_test_target.append(random.choice(list(filter(lambda l: l != y_test[i], labels))))
        y_test_target = np.array(y_test_target)

        # If requested: Reduce dimensionality
        X_train_orig = np.copy(X_train)
        X_test_orig = np.copy(X_test)
        projection_matrix = None
        projection_mean_sub = None
        pca = None
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim)
            pca.fit(X_train)

            projection_matrix = pca.components_ # Projection matrix
            projection_mean_sub = pca.mean_

            X_train = np.dot(X_train - projection_mean_sub, projection_matrix.T)
            X_test = np.dot(X_test - projection_mean_sub, projection_matrix.T)

        # Fit classifier
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
        if use_decision_tree is True:
            model = DecisionTreeClassifier(max_depth=7, random_state=42)
        model.fit(X_train, y_train)

        # Compute accuracy on test set
        print("Accuracy: {0}".format(accuracy_score(y_test, model.predict(X_test))))

        # For each class, fit density estimators
        density_estimators = {}
        kernel_density_estimators = {}
        labels = np.unique(y)
        for label in labels:
            # Get all samples with the 'correct' label
            idx = y_train == label
            X_ = X_train[idx, :]

            # Optimize hyperparameters
            cv = GridSearchCV(estimator=KernelDensity(), iid=False, param_grid={'bandwidth': np.arange(0.1, 10.0, 0.05)}, n_jobs=-1, cv=5)
            cv.fit(X_)
            bandwidth = cv.best_params_["bandwidth"]
            print("bandwidth: {0}".format(bandwidth))

            cv = GridSearchCV(estimator=GaussianMixture(covariance_type='full'), iid=False, param_grid={'n_components': range(2, 10)}, n_jobs=-1, cv=5)
            cv.fit(X_)
            n_components = cv.best_params_["n_components"]
            print("n_components: {0}".format(n_components))

            # Build density estimators
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(X_)

            de = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
            de.fit(X_)

            density_estimators[label] = de
            kernel_density_estimators[label] = kde

        # For each point in the test set
        # Compute and plot counterfactual without density constraints
        print("n_test_samples: {0}".format(X_test.shape[0]))
        for i in range(X_test.shape[0]):
            x_orig = X_test[i,:]
            x_orig_orig = X_test_orig[i,:]
            y_orig = y_test[i]
            y_target = y_test_target[i]

            if(model.predict([x_orig]) == y_target):  # Model already predicts target label!
                print("Requested prediction already satisfied")
                continue

            # Compute and plot counterfactual WITH kernel density constraints
            idx = y_train == y_target
            X_ = X_train[idx, :]

            # Build density estimator
            de = density_estimators[y_target]
            kde = kernel_density_estimators[y_target]

            # Compute media NLL of training samples
            # TODO: Move this to the outer loop
            from scipy.stats import multivariate_normal
            densities_training_samples = []
            densities_training_samples_ex = []
            for j in range(X_.shape[0]):
                x = X_[j,:]
                z = []
                dim = x.shape[0]
                for i in range(de.weights_.shape[0]):
                    x_i = de.means_[i]
                    w_i = de.weights_[i]
                    cov = de.covariances_[i]
                    cov = np.linalg.inv(cov)

                    b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))
                    z.append(np.dot(x - x_i, np.dot(cov, x - x_i)) + b) # NLL

                densities_training_samples.append(np.min(z))
                densities_training_samples_ex.append(z)

            densities_training_samples = np.array(densities_training_samples)
            densities_training_samples_ex = np.array(densities_training_samples_ex)

            # Compute soft cluster assignments
            cluster_prob_ = de.predict_proba(X_)
            density_threshold = np.median(densities_training_samples)
            # Compute high density ellipsoids - constraint: test if sample is included in ellipsoid -> this is the same as the proposed constraint but nummerically much more stable, in particular when we add a dimensionality reduction from a high dimensional space to a low dimensional space
            r = HighDensityEllipsoids(X_, densities_training_samples_ex, cluster_prob_, de.means_, de.covariances_, density_threshold).compute_ellipsoids()
            
            # Compute counterfactual without any density/plausibility/feasibility constraints
            xcf_t1 = time.time()
            cf = None
            if use_decision_tree is False:
                cf = PlausibleCounterfactualOfHyperplaneClassifier(model.coef_, model.intercept_, n_dims=X_train.shape[1], density_constraint=False, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub)
            else:
                cf = PlausibleCounterfactualOfDecisionTree(model, n_dims=X_train.shape[1], density_constraint=False, ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub)
            xcf = cf.compute_counterfactual(x_orig_orig, y=y_target)
            xcf_t1 = time.time() - xcf_t1
            if xcf is None:
                print("No counterfactual found!")
                continue

            # Compute counterfactul with proposed density constraint
            xcf_t2 = time.time()
            cf2 = None
            if use_decision_tree is False:
                cf2 = PlausibleCounterfactualOfHyperplaneClassifier(model.coef_, model.intercept_, n_dims=X_train.shape[1], ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
            else:
                cf2 = PlausibleCounterfactualOfDecisionTree(model, n_dims=X_train.shape[1], ellipsoids_r=r, gmm_weights=de.weights_, gmm_means=de.means_, gmm_covariances=de.covariances_, projection_matrix=projection_matrix, projection_mean_sub=projection_mean_sub, density_threshold=density_threshold)
            xcf2 = cf2.compute_counterfactual(x_orig_orig, y=y_target)
            xcf_t2 = time.time() - xcf_t2
            if xcf2 is None:
                print("No counterfactual found!")
                continue

            original_data.append(x_orig_orig)
            original_data_labels.append(y_orig)
            cfs_with_density_constraint.append(xcf2)
            cfs_without_density_constraint.append(xcf)
            cfs_target_label.append(y_target)
            computation_time_without_density_constraint.append(xcf_t1)
            computation_time_with_density_constraint.append(xcf_t2)
            distances_with_density_constraint.append(np.sum(np.abs(x_orig_orig - xcf2)))
            distances_without_density_constraint.append(np.sum(np.abs(x_orig_orig - xcf)))

            if pca is not None: # If necessary: Project the counterfactuals to the lower dimensional space where we did the density estimation
                xcf = pca.transform([xcf])
                xcf2 = pca.transform([xcf2])

            # Evaluate
            scores_without_density_constraint.append(kde.score_samples(xcf.reshape(1, -1)))
            scores_with_density_constraint.append(kde.score_samples(xcf2.reshape(1, -1)))

    # Final evaluation
    print("Without density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_without_density_constraint), np.mean(scores_without_density_constraint), np.var(scores_without_density_constraint)))
    print("With density constrain: Median: {0} Mean: {1} Var: {2}".format(np.median(scores_with_density_constraint), np.mean(scores_with_density_constraint), np.var(scores_with_density_constraint)))
    
    print("Computation time: With density constraint: {0} Without density constraint: {1}".format(np.median(computation_time_with_density_constraint), np.median(computation_time_without_density_constraint)))
    print("Distances: With density constraint: {0} {1} Without density constraint: {2} {3}".format(np.median(distances_with_density_constraint), np.mean(distances_with_density_constraint), np.median(distances_without_density_constraint), np.mean(distances_without_density_constraint)))

    #"""
    # Plot some samples: Counterfactual generated with vs. without density constraint
    original_data = np.array(original_data)
    original_data_labels = np.array(original_data_labels)
    cfs_with_density_constraint = np.array(cfs_with_density_constraint)
    cfs_without_density_constraint = np.array(cfs_without_density_constraint)
    cfs_target_label = np.array(cfs_target_label)
    np.savez("cfs_comparision_data_softmax_regression", X_original=original_data, y_original=original_data_labels, y_target=cfs_target_label, X_with_density_constraint=cfs_with_density_constraint, X_without_density_constraint=cfs_without_density_constraint)
    #"""
