# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from .tree import get_leafs_from_tree


class PlausibleCounterfactualOfHyperplaneClassifier:
    def __init__(
        self,
        w,
        b,
        n_dims,
        ellipsoids_r,
        gmm_weights,
        gmm_means,
        gmm_covariances,
        projection_matrix=None,
        projection_mean_sub=None,
        density_constraint=True,
        density_threshold=-85,
    ):
        self.w = w
        self.b = b

        self.n_dims = n_dims
        self.gmm_weights = gmm_weights
        self.gmm_means = gmm_means
        self.gmm_covariances = gmm_covariances
        self.ellipsoids_r = ellipsoids_r
        self.projection_matrix = (
            np.eye(self.n_dims) if projection_matrix is None else projection_matrix
        )
        self.projection_mean_sub = (
            np.zeros(self.n_dims)
            if projection_mean_sub is None
            else projection_mean_sub
        )
        self.density_constraint = density_constraint

        self.min_density = density_threshold
        self.epsilon = 1e-3
        self.gmm_cluster_index = 0  # For internal use only!

    def _build_constraints(self, var_x, y):
        constraints = []
        if self.w.shape[0] > 1:
            for i in range(self.w.shape[0]):
                if i != y:
                    constraints += [
                        (self.projection_matrix @ (var_x - self.projection_mean_sub)).T
                        @ (self.w[i, :] - self.w[y, :])
                        + (self.b[i] - self.b[y])
                        + self.epsilon
                        <= 0
                    ]
        else:
            if y == 0:
                return [
                    (self.projection_matrix @ (var_x - self.projection_mean_sub)).T
                    @ self.w.reshape(-1, 1)
                    + self.b
                    + self.epsilon
                    <= 0
                ]
            else:
                return [
                    (self.projection_matrix @ (var_x - self.projection_mean_sub)).T
                    @ self.w.reshape(-1, 1)
                    + self.b
                    - self.epsilon
                    >= 0
                ]

        return constraints

    def compute_counterfactual(self, x, y, regularizer="l1"):
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])

        xcf = None
        s = float("inf")
        for i in range(self.gmm_weights.shape[0]):
            try:
                self.gmm_cluster_index = i
                xcf_ = self.build_solve_opt(x, y, mad)
                if xcf_ is None:
                    continue

                s_ = None
                if regularizer == "l1":
                    s_ = np.sum(np.abs(xcf_ - x))
                else:
                    s_ = np.linalg.norm(xcf_ - x, ord=2)

                if s_ <= s:
                    s = s_
                    xcf = xcf_
            except Exception as ex:
                print(ex)
        return xcf

    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self, x_orig, y, mad=None):
        dim = x_orig.shape[0]

        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)

        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)  # noqa: E741

        # Construct constraints
        constraints = self._build_constraints(x, y)

        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[i]
            cov = self.gmm_covariances[i]
            cov = np.linalg.inv(cov)
            """
            w_i = self.gmm_weights[i]
            b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))
            constraints += [cp.quad_form((self.projection_matrix @ (x - self.projection_mean_sub)) - x_i, cov) + b <= self.min_density]
            """
            constraints += [
                cp.quad_form(
                    self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov
                )
                - self.ellipsoids_r[i]
                <= 0
            ]  # Numerically much more stable than the explicit density component constraint

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1.0 / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)  # Minimize (weighted) Manhattan distance
            constraints += [
                Upsilon @ (x - x_orig) <= beta,
                (-1.0 * Upsilon) @ (x - x_orig) <= beta,
                I @ beta >= z,
            ]
        else:
            f = cp.Minimize(
                (1 / 2) * cp.quad_form(x, I) - x_orig.T @ x
            )  # Minimize L2 distance

        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return x.value


class PlausibleCounterfactualOfDecisionTree:
    def __init__(
        self,
        model,
        n_dims,
        ellipsoids_r,
        gmm_weights,
        gmm_means,
        gmm_covariances,
        projection_matrix=None,
        projection_mean_sub=None,
        density_constraint=True,
        density_threshold=-85,
    ):
        self.model = model

        self.n_dims = n_dims
        self.gmm_weights = gmm_weights
        self.gmm_means = gmm_means
        self.gmm_covariances = gmm_covariances
        self.ellipsoids_r = ellipsoids_r
        self.projection_matrix = (
            np.eye(self.n_dims) if projection_matrix is None else projection_matrix
        )
        self.projection_mean_sub = (
            np.zeros(self.n_dims)
            if projection_mean_sub is None
            else projection_mean_sub
        )
        self.density_constraint = density_constraint

        self.min_density = density_threshold
        self.epsilon = 1e-3
        self.gmm_cluster_index = 0  # For internal use only

    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def _build_constraints(self, var_x, y_target, path_to_leaf):
        constraints = []
        eps = 1.0e-5

        for j in range(0, len(path_to_leaf) - 1):
            feature_id = path_to_leaf[j][1]
            threshold = path_to_leaf[j][2]
            direction = path_to_leaf[j][3]

            if direction == "<":
                constraints.append(
                    (self.projection_matrix @ var_x)[feature_id] + eps <= threshold
                )
            elif direction == ">":
                constraints.append(
                    (self.projection_matrix @ var_x)[feature_id] - eps >= threshold
                )

        return constraints

    def build_solve_opt(self, x_orig, y, path_to_leaf, mad=None):
        dim = x_orig.shape[0]

        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)

        # Constants
        c = np.ones(dim)
        z = np.zeros(dim)
        I = np.eye(dim)  # noqa: E741

        # Construct constraints
        constraints = self._build_constraints(x, y, path_to_leaf)

        if self.density_constraint is True:
            i = self.gmm_cluster_index
            x_i = self.gmm_means[i]
            cov = self.gmm_covariances[i]
            cov = np.linalg.inv(cov)

            """
            w_i = self.gmm_weights[i]
            b = b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))
            constraints += [cp.quad_form(self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov) + b <= self.min_density]
            """
            constraints += [
                cp.quad_form(
                    self.projection_matrix @ (x - self.projection_mean_sub) - x_i, cov
                )
                - self.ellipsoids_r[i]
                <= 0
            ]

        # If necessary, construct the weight matrix for the weighted Manhattan distance
        Upsilon = None
        if mad is not None:
            alpha = 1.0 / mad
            Upsilon = np.diag(alpha)

        # Build the final program
        f = None
        if mad is not None:
            f = cp.Minimize(c.T @ beta)  # Minimize (weighted) Manhattan distance
            constraints += [
                Upsilon @ (x - x_orig) <= beta,
                (-1.0 * Upsilon) @ (x - x_orig) <= beta,
                I @ beta >= z,
            ]
        else:
            f = cp.Minimize(
                (1 / 2) * cp.quad_form(x, I) - x_orig.T @ x
            )  # Minimize L2 distance

        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return x.value

    def compute_counterfactual(self, x, y, regularizer="l1"):
        mad = None
        if regularizer == "l1":
            mad = np.ones(x.shape[0])

        xcf = None
        s = float("inf")

        # Find all paths that lead to a valid (but not necessarily feasible) counterfactual
        # Collect all leafs
        leafs = get_leafs_from_tree(self.model.tree_, classifier=True)

        # Filter leafs for predictions
        leafs = list(filter(lambda x: x[-1][2] == y, leafs))

        if len(leafs) == 0:
            raise ValueError(
                "Tree does not has a path/leaf yielding the requested outcome specified in 'y_target'"
            )

        # For each leaf: Compute feasible counterfactual
        # TODO: Make this more efficient!
        for path_to_leaf in leafs:
            for i in range(self.gmm_weights.shape[0]):
                try:
                    self.gmm_cluster_index = i
                    xcf_ = self.build_solve_opt(x, y, path_to_leaf, mad)
                    if xcf_ is None:
                        continue

                    s_ = None
                    if regularizer == "l1":
                        s_ = np.sum(np.abs(xcf_ - x))
                    else:
                        s_ = np.linalg.norm(xcf_ - x, ord=2)

                    if s_ <= s:
                        s = s_
                        xcf = xcf_
                except Exception as ex:
                    print(ex)
        return xcf


class HighDensityEllipsoids:
    def __init__(
        self, X, X_densities, cluster_probs, means, covariances, density_threshold=None
    ):
        self.X = X
        self.X_densities = X_densities
        self.density_threshold = (
            density_threshold if density_threshold is not None else float("-inf")
        )
        self.cluster_probs = cluster_probs
        self.means = means
        self.covariances = covariances
        self.t = 0.9
        self.epsilon = 0  # 1e-5

    def compute_ellipsoids(self):
        return self.build_solve_opt()

    def _solve(self, prob):
        prob.solve(solver=cp.SCS, verbose=False)

    def build_solve_opt(self):
        n_ellipsoids = self.cluster_probs.shape[1]
        n_samples = self.X.shape[0]

        # Variables
        r = cp.Variable(n_ellipsoids, pos=True)

        # Construct constraints
        constraints = []
        for i in range(n_ellipsoids):
            mu_i = self.means[i]
            cov_i = np.linalg.inv(self.covariances[i])

            for j in range(n_samples):
                if (
                    self.X_densities[j][i] >= self.density_threshold
                ):  # At least as good as a requested NLL
                    x_j = self.X[j, :]

                    a = x_j - mu_i
                    b = np.dot(a, np.dot(cov_i, a))
                    constraints.append(b <= r[i])

        # Build the final program
        f = cp.Minimize(cp.sum(r))
        prob = cp.Problem(f, constraints)

        # Solve it!
        self._solve(prob)

        return r.value
