"""
This module contains the implementation an evolutionary algorithm for
computing multi-instance counterfactual explanations --  see Artelt et al. [2024] for details.
"""

from __future__ import annotations
from typing import Any
import random
from copy import deepcopy
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.constraints.as_penalty import ConstraintsAsPenalty


class MixedVarGroupCfProblem(ElementwiseProblem):
    def __init__(
        self,
        X_orig,
        y_target,
        clf,
        features_type,
        features_range,
        features_idx_whitelist,
    ):
        self.X_orig = X_orig
        self.y_target = y_target
        self.features_type = features_type
        self.old_features_range = deepcopy(features_range)
        self.features_range = features_range
        self.clf = clf
        self.features_idx_whitelist = features_idx_whitelist

        for (
            f_idx
        ) in self.features_idx_whitelist:  # Compute thight bounds of feasible changes
            if self.features_type == "real":
                self.features_range[f_idx] = (
                    self.features_range[f_idx][0] - min(self.X_orig[:, f_idx]),
                    self.features_range[f_idx][1] - max(self.X_orig[:, f_idx]),
                )
            elif self.features_type == "int":
                self.features_range[f_idx] = (
                    self.features_range[f_idx][0] - int(min(self.X_orig[:, f_idx])),
                    self.features_range[f_idx][1] - int(max(self.X_orig[:, f_idx])),
                )

        super().__init__(
            n_var=len(self.features_idx_whitelist),
            n_obj=1,
            n_ieq_constr=len(self.X_orig),
        )

    def complete_sample(self, delta_cf):
        delta_cf_ = np.array(
            [-1 for _ in range(self.X_orig.shape[1])]
        )  # All categorical variables!

        if len(self.features_idx_whitelist) != self.X_orig.shape[1]:
            for i, j in zip(
                self.features_idx_whitelist, range(len(self.features_idx_whitelist))
            ):  # Complete sample (i.e. not all features are be mutable!)
                delta_cf_[i] = delta_cf[j]
        else:
            for i in range(
                len(delta_cf)
            ):  # Copy values over if every feature is mutable
                delta_cf_[i] = delta_cf[i]

        return delta_cf_

    def apply_changes(self, delta_cf, X):
        X_ = np.copy(X)

        for i in range(len(self.features_type)):
            if self.features_type[i] == "cat" and i in self.features_idx_whitelist:
                if delta_cf[i] != -1:  # Leave the variable unchanged?
                    X_[:, i] = delta_cf[i]
            else:
                X_[:, i] += delta_cf[i]

        return X_

    def _evaluate(self, delta_cf, out, *args, **kwargs):
        delta_cf_ = self.complete_sample(
            delta_cf
        )  # Complete sample if necessary (i.e. not all features might be mutable!)
        X_orig_ = self.apply_changes(delta_cf_, self.X_orig)

        loss = 0.0
        loss_numbers = []
        for i, f_type in zip(range(len(self.features_type)), self.features_type):
            if delta_cf_[i] != 0 and delta_cf_[i] != -1:
                if f_type == "cat":
                    loss += 1
                elif f_type == "int":
                    loss_numbers.append(np.abs(delta_cf_[i]))
                elif f_type == "real":
                    loss_numbers.append(np.abs(delta_cf_[i]))
        if len(loss_numbers) != 0:
            loss += np.sum(
                loss_numbers / max(loss_numbers)
            )  # Scale other losses and add them to the final loss

        out["F"] = loss
        out["G"] = (
            0.5 - self.clf.predict_proba(X_orig_)[:, self.y_target]
        )  # One constraint for each sample


##########################################################################################################################
# https://pymoo.org/customization/custom.html


class MySampling(Sampling):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, len(problem.features_idx_whitelist)))

        for i in range(n_samples):
            for j, f_idx in zip(
                range(len(problem.features_idx_whitelist)),
                problem.features_idx_whitelist,
            ):
                val_range = problem.features_range[f_idx]

                if problem.features_type[f_idx] == "cat":
                    X[i, j] = random.choice(val_range + [-1])
                elif problem.features_type[f_idx] == "int":
                    X[i, j] = np.random.randint(low=val_range[0], high=val_range[1])
                elif problem.features_type[f_idx] == "real":
                    X[i, j] = np.random.uniform(low=val_range[0], high=val_range[1])
                    X[i, j] = np.round(X[i, j], decimals=2)

        return X


class MyMutation(Mutation):
    def __init__(self, feature_mut_prob=0.2):
        self.feature_mut_prob = (
            feature_mut_prob  # With probability of 80% - apply a mutation to a feature
        )
        super().__init__()

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            for j, f_idx in zip(
                range(len(problem.features_idx_whitelist)),
                problem.features_idx_whitelist,
            ):
                r = np.random.random()
                if r < self.feature_mut_prob:
                    continue

                val_range = problem.features_range[f_idx]
                if problem.features_type[f_idx] == "cat":
                    X[i, j] = random.choice(val_range + [-1])
                elif problem.features_type[f_idx] == "int":
                    X[i, j] = np.random.randint(low=val_range[0], high=val_range[1])
                elif problem.features_type[f_idx] == "real":
                    X[i, j] += np.random.normal(
                        loc=0.0, scale=val_range[1] - val_range[0]
                    )
                    X[i, j] = max(
                        val_range[0], X[i, j]
                    )  # Make sure value does not leave range of feasible values
                    X[i, j] = min(val_range[1], X[i, j])
                    X[i, j] = np.round(X[i, j], decimals=2)  # Round floats

        return X


class MyDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return np.all(a.X == b.X)


####################################################################################################################################


def compute_mixedvar_groupcf(
    X_orig: np.ndarray,
    y_target: int,
    clf: Any,
    features_type: list[str],
    features_range: list[tuple[float, float]],
    features_idx_whitelist: list[int],
) -> tuple[np.ndarray, float]:
    """
    Computes a multi-instance counterfactual explanation of a given set of instances and classifier.

    Parameters
    ----------
    X_orig : `numpy.ndarray`
        Set (2d-array) of instances for which a multi-instance counterfactual is to be computed --
        note that are all instances must be classified the same!
    y_target : `int`
        Target label -- assuming a probabilistic classifier (i.e. class-wise probabilities are predicted).
    clf : `Any`
        Probabilistic classifier.

        Must implement a `predict_proba` method that returns class-wise probabilities.
    features_type : `list[str]`
        Type of each feature -- supported types are: 'real', 'int', and 'cat'.
    features_range : `list[tuple[float, float]]`
        Value ranges of each feature.
    features_idx_whitelist : `list[int]`
        List of feature indices that are mutable (i.e. can be changed in the multi-instance counterfactual).

    Returns
    -------
    `tuple[np.ndarray, float]`
        Multi-instance counterfactual explanation together with the error rate
        (i.e. percentage of instances for which the counterfactual is not valid).
    """
    algo = GA(
        pop_size=400,
        sampling=MySampling(),
        crossover=UniformCrossover(),
        mutation=MyMutation(),
        eliminate_duplicates=MyDuplicateElimination(),
    )

    problem = MixedVarGroupCfProblem(
        X_orig, y_target, clf, features_type, features_range, features_idx_whitelist
    )
    res = minimize(ConstraintsAsPenalty(problem, penalty=100.0), algo, verbose=False)

    if len(res.X.shape) == 2:
        delta_cf = res.X[np.argsort(res.F)][0].flatten()
    else:
        delta_cf = res.X

    # Evaluate solution
    delta_cf_ = problem.complete_sample(
        delta_cf
    )  # Complete sample if necessary (i.e. not all features might be mutable!
    X_cf = problem.apply_changes(delta_cf_, X_orig)
    y_cf_pred = clf.predict(X_cf)

    err_rate = 0
    for i in range(len(X_cf)):
        if y_cf_pred[i] != y_target:
            err_rate += 1
    err_rate /= len(X_cf)

    if err_rate == 1.0:
        raise RuntimeError(
            "Computation of multi-instance counterfactual explanation failed"
        )

    return delta_cf_, err_rate
