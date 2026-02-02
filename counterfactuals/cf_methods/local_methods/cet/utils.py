import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch
from lingam import DirectLiNGAM
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.interpolate import interp1d
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde as kde
from scipy.stats import median_abs_deviation as mad
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


def flatten(x):
    return sum(x, [])


def supp(a, tol=1e-8):
    return np.where(abs(a) > tol)[0]


def greedy_select_protos(K, candidate_indices, m, is_K_sparse=False):
    # From https://github.com/BeenKim/MMD-critic/blob/master/mmd.py
    import sys

    if len(candidate_indices) != np.shape(K)[0]:
        K = K[:, candidate_indices][candidate_indices, :]
    n = len(candidate_indices)
    if is_K_sparse:
        colsum = 2 * np.array(K.sum(0)).ravel() / n
    else:
        colsum = 2 * np.sum(K, axis=0) / n
    selected = np.array([], dtype=int)
    value = np.array([])
    for i in range(m):
        maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)
        s1array = colsum[candidates]
        if len(selected) > 0:
            temp = K[selected, :][:, candidates]
            if is_K_sparse:
                s2array = temp.sum(0) * 2 + K.diagonal()[candidates]
            else:
                s2array = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]
            s2array = s2array / (len(selected) + 1)
            s1array = s1array - s2array
        else:
            if is_K_sparse:
                s1array = s1array - (np.abs(K.diagonal()[candidates]))
            else:
                s1array = s1array - (np.abs(np.diagonal(K)[candidates]))
        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)
        KK = K[selected, :][:, selected]
        if is_K_sparse:
            KK = KK.todense()
    return candidate_indices[selected]


def prototype_selection(X, subsample=20, kernel="rbf"):
    return (
        greedy_select_protos(
            pairwise_kernels(X, metric=kernel), np.array(range(X.shape[0])), subsample
        )
        if subsample > 1
        else np.array(range(X.shape[0]))
    )


def submodular_picking(X, budget=4):
    D = len(X)
    X_supp = [(i, supp(x)) for i, x in enumerate(X)]
    I = []
    supps = np.array([])
    while len(I) < budget:
        n = np.argmax([np.union1d(supps, cand[1]).shape[0] for cand in X_supp])
        i_n, supp_n = X_supp.pop(n)
        supps = np.union1d(supps, supp_n)
        I += [i_n]
    return I


def CumulativeDistributionFunction(x_d, X_d, l_buff=1e-6, r_buff=1e-6):
    kde_estimator = kde(X_d)
    pdf = kde_estimator(x_d)
    cdf_raw = np.cumsum(pdf)
    total = cdf_raw[-1] + l_buff + r_buff
    cdf = (l_buff + cdf_raw) / total
    percentile_ = interp1d(
        x=x_d,
        y=cdf,
        copy=False,
        fill_value=(l_buff, 1.0 - r_buff),
        bounds_error=False,
        assume_sorted=False,
    )
    return percentile_


def interaction_matrix(
    X,
    interaction_type="causal",
    prior_knowledge=None,
    measure="pwling",
    estimator="ML",
    file_name="",
):
    if interaction_type == "causal":
        lingam = DirectLiNGAM(prior_knowledge=prior_knowledge, measure=measure).fit(X)
        B = lingam.adjacency_matrix_
        C = np.zeros([X.shape[1], X.shape[1]])
        for d in range(1, X.shape[1]):
            C += np.linalg.matrix_power(B, d)
        return B, C
    elif interaction_type == "correlation":
        return np.corrcoef(X.T) - np.eye(X.shape[1])
    elif interaction_type == "covariance":
        if estimator == "ML":
            est = EmpiricalCovariance(store_precision=True, assume_centered=False).fit(X)
        elif estimator == "MCD":
            est = MinCovDet(store_precision=True, assume_centered=False, support_fraction=None).fit(
                X
            )
        cov = est.covariance_
        if np.linalg.matrix_rank(cov) != X.shape[1]:
            cov += 1e-6 * np.eye(X.shape[1])
        l_, P_ = np.linalg.eig(np.linalg.inv(cov))
        l = np.diag(np.sqrt(l_))
        P = P_.T
        U = P.T.dot(l).T
        return cov, U
    elif interaction_type == "precomputed":
        df = pd.read_csv(file_name)
        return df.values


class LimeEstimator:
    def __init__(
        self,
        mdl,
        X,
        n_samples=10000,
        feature_types=[],
        feature_categories=[],
        alpha=1.0,
    ):
        self.mdl_ = mdl
        self.mdl_local_ = Ridge(alpha=alpha)
        self.N_, self.D_ = X.shape
        self.n_samples_ = n_samples
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        self.feature_types_ = (
            feature_types if len(feature_types) == self.D_ else ["C" for d in range(self.D_)]
        )
        self.feature_category_ = feature_categories
        self.feature_category_flatten_ = flatten(feature_categories)
        self.feature_ordered_ = [
            d for d in range(self.D_) if feature_types[d] == "C" or feature_types[d] == "I"
        ]
        self.feature_binary_ = [
            d
            for d in range(self.D_)
            if feature_types[d] == "B" and d not in self.feature_category_flatten_
        ]

    def getNeighbors(self, x):
        N_x = np.zeros([self.n_samples_, self.D_])
        for d in self.feature_ordered_:
            if self.feature_types_[d] == "I":
                N_x[:, d] = np.random.normal(x[d], self.std_[d], self.n_samples_).astype(int)
            else:
                N_x[:, d] = np.random.normal(x[d], self.std_[d], self.n_samples_)
        for d in self.feature_binary_:
            N_x[:, d] = (np.random.uniform(0, 1, self.n_samples_) <= self.mean_[d]).astype(int)
        for G in self.feature_category_:
            cats = np.random.choice(G, self.n_samples_, p=self.mean_[G])
            for n, d in enumerate(cats):
                N_x[n, d] = 1
        N_x = np.concatenate([x.reshape(1, -1), N_x], axis=0)
        return N_x

    def getWeights(self, x, N_x):
        distance = pairwise_distances(N_x / self.std_, (x / self.std_).reshape(1, -1)).reshape(-1)
        kernel_width = np.sqrt(self.D_) * 0.75
        weights = np.sqrt(np.exp(-(distance**2) / kernel_width**2))
        return weights

    def fit(self, x, target_label=None):
        N_x = self.getNeighbors(x)
        weights = self.getWeights(x, N_x)
        if target_label is None:
            target_label = int(1 - self.mdl_.predict(x.reshape(1, -1)))
        self.mdl_local_ = self.mdl_local_.fit(
            N_x, self.mdl_.predict_proba(N_x)[:, target_label], sample_weight=weights
        )
        self.offset_ = (
            self.mdl_.predict_proba(x.reshape(1, -1))[0, target_label]
            - self.mdl_local_.predict(x.reshape(1, -1))[0]
        )
        return self

    def approximate(self, x):
        self = self.fit(x)
        return self.mdl_local_.coef_, self.mdl_local_.intercept_ + self.offset_ - 0.5

    def predict(self, X):
        return self.mdl_local_.predict(X)


class MyTabNetClassifier:
    def __init__(
        self,
        feature_types,
        pretraining_ratio=0.5,
        max_epochs=1000,
        patience=50,
        class_weight="uniform",
        verbose=0,
    ):
        cat_idxs = [d for d in range(len(feature_types)) if feature_types[d] == "B"]
        cat_dims = [2] * len(cat_idxs)
        self.pretrainer = TabNetPretrainer(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            cat_emb_dim=1,
            mask_type="sparsemax",
            verbose=verbose,
        )
        self.classifier = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            cat_emb_dim=1,
            mask_type="sparsemax",
            verbose=verbose,
        )
        self.pretraining_ratio = pretraining_ratio
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = int(class_weight == "balanced")
        self.verbose = bool(verbose)

    def fit(self, X, y, X_vl=None, y_vl=None, eval_metric=["auc"]):
        eval_set_pre = [X] if X_vl is None else [X, X_vl]
        eval_set = [(X, y)] if X_vl is None else [(X, y), (X_vl, y_vl)]
        eval_name = ["train"] if X_vl is None else ["train", "validation"]

        if self.verbose:
            self.pretrainer.fit(
                X_train=X,
                eval_set=eval_set_pre,
                eval_name=eval_name,
                pretraining_ratio=self.pretraining_ratio,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=1024,
                virtual_batch_size=128,
            )
            self.classifier.fit(
                X_train=X,
                y_train=y,
                eval_set=eval_set,
                eval_name=eval_name,
                eval_metric=eval_metric,
                max_epochs=self.max_epochs,
                patience=self.patience,
                weights=self.weights,
                from_unsupervised=self.pretrainer,
                batch_size=1024,
                virtual_batch_size=128,
            )
        else:
            with redirect_stdout(open(os.devnull, "w")):
                self.pretrainer.fit(
                    X_train=X,
                    eval_set=eval_set_pre,
                    eval_name=eval_name,
                    pretraining_ratio=self.pretraining_ratio,
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    batch_size=1024,
                    virtual_batch_size=128,
                )
                self.classifier.fit(
                    X_train=X,
                    y_train=y,
                    eval_set=eval_set,
                    eval_name=eval_name,
                    eval_metric=eval_metric,
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    weights=self.weights,
                    from_unsupervised=self.pretrainer,
                    batch_size=1024,
                    virtual_batch_size=128,
                )
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        return (y == self.predict(X)).mean()


class Cost:
    def __init__(
        self,
        X=[],
        Y=[],
        feature_types=[],
        feature_categories=[],
        feature_constraints=[],
        max_candidates=50,
        tol=1e-6,
    ):
        self.X_ = X
        self.Y_ = Y
        self.N_, self.D_ = X.shape
        self.feature_types_ = (
            feature_types if len(feature_types) == self.D_ else ["C" for d in range(self.D_)]
        )
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = (
            feature_constraints
            if len(feature_constraints) == self.D_
            else ["" for d in range(self.D_)]
        )
        self.tol_ = tol

        self.X_lb_, self.X_ub_ = X.min(axis=0), X.max(axis=0)
        self.steps_ = [
            (self.X_ub_[d] - self.X_lb_[d]) / max_candidates if self.feature_types_[d] == "C" else 1
            for d in range(self.D_)
        ]
        self.grids_ = [
            np.arange(self.X_lb_[d], self.X_ub_[d] + self.steps_[d], self.steps_[d])
            for d in range(self.D_)
        ]

        self.Q_ = None
        self.weights_ = None

    def getFeatureWeight(self, cost_type="uniform"):
        weights = np.ones(self.D_)
        if cost_type == "MAD":
            for d in range(self.D_):
                weight = mad(self.X_[:, d], scale="normal")
                if self.feature_types_[d] == "B" or abs(weight) < self.tol_:
                    weights[d] = (self.X_[:, d] * 1.4826).std()
                else:
                    weights[d] = weight**-1
        elif cost_type == "standard":
            weights = np.std(self.X_, axis=0) ** -1
        elif cost_type == "normalize":
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif cost_type == "robust":
            q25, q75 = np.percentile(self.X_, [0.25, 0.75], axis=0)
            for d in range(self.D_):
                if q75[d] - q25[d] == 0:
                    weights[d] = self.tol_**-1
                else:
                    weights = (q75[d] - q25) ** -1
        return weights

    def compute(self, x, a, cost_type="TLPS"):
        cost = 0.0
        if cost_type == "TLPS" or cost_type == "MPS":
            if self.Q_ is None:
                self.Q_ = [
                    None
                    if self.feature_constraints_[d] == "FIX"
                    else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d])
                    for d in range(self.D_)
                ]
            for d in range(self.D_):
                if self.Q_[d] != None:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    if cost_type == "TLPS":
                        cost += abs(np.log2((1 - Q_d(x[d] + a[d])) / (1 - Q_0)))
                    else:
                        if d in flatten(self.feature_categories_) and a[d] < 0:
                            continue
                        c = abs(Q_d(x[d] + a[d]) - Q_0)
                        if cost < c:
                            cost = c
        else:
            if self.weights_ is None:
                self.weights_ = self.getFeatureWeight(cost_type=cost_type)
            p = 2 if cost_type == "PCC" else 1
            for d in range(self.D_):
                cost += self.weights_[d] * (abs(a[d]) ** p)
        return cost


class Action:
    def __init__(
        self,
        x,
        a,
        scores={},
        target_name="Output",
        target_labels=["Good", "Bad"],
        label_before=1,
        label_after=0,
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        print_instance=False,
    ):
        self.x_ = x
        self.a_ = a
        self.scores_ = scores
        self.target_name_ = target_name
        self.labels_ = [target_labels[label_before], target_labels[label_after]]
        self.feature_names_ = (
            feature_names
            if len(feature_names) == len(x)
            else ["x_{}".format(d) for d in range(len(x))]
        )
        self.feature_types_ = (
            feature_types if len(feature_types) == len(x) else ["C" for d in range(len(x))]
        )
        self.feature_categories_ = feature_categories
        self.print_instance = print_instance

        self.feature_categories_inv_ = []
        for d in range(len(x)):
            g = -1
            if self.feature_types_[d] == "B":
                for i, cat in enumerate(self.feature_categories_):
                    if d in cat:
                        g = i
                        break
            self.feature_categories_inv_.append(g)

    def __str__(self):
        if self.a_ is None:
            return "No Feasible Action\n"
        s = ""
        if self.print_instance:
            s += "* Instance:\n"
            for d, x_d in enumerate(self.x_):
                g = self.feature_categories_inv_[d]
                if g == -1:
                    if self.feature_types_[d] == "C":
                        s += "\t* {}: {.4f}\n".format(self.feature_names_[d], x_d)
                    if self.feature_types_[d] == "B":
                        s += "\t* {}: {}\n".format(self.feature_names_[d], bool(x_d))
                    else:
                        s += "\t* {}: {}\n".format(self.feature_names_[d], int(x_d))
                else:
                    if x_d != 1:
                        continue
                    s += "\t* {}\n".format(self.feature_names_[d])
        s += "* Action ({}: {} -> {}):\n".format(
            self.target_name_, self.labels_[0], self.labels_[1]
        )
        i = 0
        for i, d in enumerate(np.where(abs(self.a_) > 1e-8)[0]):
            num = "*"
            g = self.feature_categories_inv_[d]
            if g == -1:
                if self.feature_types_[d] == "C":
                    s += "\t{} {}: {:.4f} -> {:.4f} ({:+.4f})\n".format(
                        num,
                        self.feature_names_[d],
                        self.x_[d],
                        self.x_[d] + self.a_[d],
                        self.a_[d],
                    )
                if self.feature_types_[d] == "B":
                    s += (
                        "\t* {}: True -> False\n".format(self.feature_names_[d])
                        if bool(self.x_[d])
                        else "\t* {}: False -> True\n".format(self.feature_names_[d])
                    )
                else:
                    s += "\t{} {}: {} -> {} ({:+})\n".format(
                        num,
                        self.feature_names_[d],
                        self.x_[d].astype(int),
                        (self.x_[d] + self.a_[d]).astype(int),
                        self.a_[d].astype(int),
                    )
            else:
                if self.x_[d] == 1:
                    continue
                cat_name, nxt = self.feature_names_[d].split(":")
                cat = self.feature_categories_[g]
                prv = self.feature_names_[cat[np.where(self.x_[cat])[0][0]]].split(":")[1]
                s += "\t{} {}: {} -> {}\n".format(num, cat_name, prv, nxt)

        if len(self.scores_) > 0:
            s += "* Scores: \n"
            for i in self.scores_.items():
                s += (
                    "\t* {0}: {1:.8f}\n".format(i[0], i[1])
                    if isinstance(i[1], float)
                    else "\t* {0}: {1}\n".format(i[0], i[1])
                )
        return s

    def a(self):
        return self.a_

    def is_feasible(self):
        return self.a_ is not None

    def sef_x(self, x):
        self.x_ = x
        return self


# class Action


ACTION_TYPES = ["B", "I", "C"]
ACTION_CONSTRAINTS = ["", "FIX", "INC", "DEC"]


class ActionCandidates:
    def __init__(
        self,
        X,
        Y=[],
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        feature_constraints=[],
        max_candidates=50,
        tol=1e-6,
    ):
        self.X_ = X
        self.Y_ = Y
        self.N_, self.D_ = X.shape
        self.feature_names_ = (
            feature_names
            if len(feature_names) == self.D_
            else ["x_{}".format(d) for d in range(self.D_)]
        )
        self.feature_types_ = (
            feature_types if len(feature_types) == self.D_ else ["C" for d in range(self.D_)]
        )
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = (
            feature_constraints
            if len(feature_constraints) == self.D_
            else ["" for d in range(self.D_)]
        )
        self.max_candidates = max_candidates
        self.tol_ = tol

        self.X_lb_, self.X_ub_ = X.min(axis=0), X.max(axis=0)
        self.steps_ = [
            (self.X_ub_[d] - self.X_lb_[d]) / max_candidates if self.feature_types_[d] == "C" else 1
            for d in range(self.D_)
        ]
        self.grids_ = [
            np.arange(self.X_lb_[d], self.X_ub_[d] + self.steps_[d], self.steps_[d])
            for d in range(self.D_)
        ]

        self.actions_ = None
        self.costs_ = None
        self.Q_ = None
        self.cov_ = None

    def getFeatureWeight(self, cost_type="uniform"):
        weights = np.ones(self.D_)
        if cost_type == "MAD":
            for d in range(self.D_):
                weight = mad(self.X_[:, d], scale="normal")
                if self.feature_types_[d] == "B" or abs(weight) < self.tol_:
                    weights[d] = (self.X_[:, d] * 1.4826).std()
                else:
                    weights[d] = weight**-1
        elif cost_type == "PCC" and len(self.Y_) == self.N_:
            for d in range(self.D_):
                weights[d] = abs(np.corrcoef(self.X_[:, d], self.Y_)[0, 1])
        elif cost_type == "standard":
            weights = np.std(self.X_, axis=0) ** -1
        elif cost_type == "normalize":
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif cost_type == "robust":
            q25, q75 = np.percentile(self.X_, [0.25, 0.75], axis=0)
            for d in range(self.D_):
                if q75[d] - q25[d] == 0:
                    weights[d] = self.tol_**-1
                else:
                    weights = (q75[d] - q25) ** -1
        return weights

    def setActionSet(self, x):
        self.actions_ = []
        for d in range(self.D_):
            if self.feature_constraints_[d] == "FIX" or self.steps_[d] < self.tol_:
                self.actions_.append(np.array([0]))
            elif self.feature_types_[d] == "B":
                if (self.feature_constraints_[d] == "INC" and x[d] == 1) or (
                    self.feature_constraints_[d] == "DEC" and x[d] == 0
                ):
                    self.actions_.append(np.array([0]))
                else:
                    self.actions_.append(np.array([1 - 2 * x[d], 0]))
            else:
                if self.feature_constraints_[d] == "INC":
                    start = x[d] + self.steps_[d]
                    stop = self.X_ub_[d] + self.steps_[d]
                elif self.feature_constraints_[d] == "DEC":
                    start = self.X_lb_[d]
                    stop = x[d]
                else:
                    start = self.X_lb_[d]
                    stop = self.X_ub_[d] + self.steps_[d]
                A_d = np.arange(start, stop, self.steps_[d]) - x[d]
                A_d = np.extract(abs(A_d) > self.tol_, A_d)
                if len(A_d) > self.max_candidates:
                    A_d = A_d[np.linspace(0, len(A_d) - 1, self.max_candidates, dtype=int)]
                A_d = np.append(A_d, 0)
                self.actions_.append(A_d)
        return self

    def setActionAndCost(self, x, y, cost_type="TLPS", p=1):
        self.costs_ = []
        self = self.setActionSet(x)
        if cost_type == "TLPS" or cost_type == "MPS":
            if self.Q_ == None:
                self.Q_ = [
                    None
                    if self.feature_constraints_[d] == "FIX"
                    else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d])
                    for d in range(self.D_)
                ]
            for d in range(self.D_):
                if self.Q_[d] == None:
                    self.costs_.append([0])
                else:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    self.costs_.append(
                        [
                            abs(np.log2((1 - Q_d(x[d] + a)) / (1 - Q_0)))
                            if cost_type == "TLPS"
                            else abs(Q_d(x[d] + a) - Q_0)
                            for a in self.actions_[d]
                        ]
                    )
        elif cost_type == "SCM" or cost_type == "DACE":
            if cost_type == "SCM":
                B_, _ = interaction_matrix(self.X_, interaction_type="causal")
                B = np.eye(self.D_) - B_
                C = self.getFeatureWeight(cost_type="standard")
            else:
                self.cov_, B = interaction_matrix(
                    self.X_[self.Y_ == y] if len(self.Y_) == self.N_ else self.X_,
                    interaction_type="covariance",
                )
                C = self.getFeatureWeight(cost_type="uniform")
            for d in range(self.D_):
                cost_d = []
                for d_ in range(self.D_):
                    cost_d.append([C[d] * B[d][d_] * a for a in self.actions_[d_]])
                self.costs_.append(cost_d)
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            if cost_type == "PCC":
                p = 2
            for d in range(self.D_):
                self.costs_.append(list(weights[d] * abs(self.actions_[d]) ** p))
        return self

    def setMultiActionSet(self, xs, union=False):
        self.actions_ = []
        for d in range(self.D_):
            if self.feature_constraints_[d] == "FIX" or self.steps_[d] < self.tol_:
                self.actions_.append(np.array([0]))
            elif self.feature_types_[d] == "B":
                x_d = xs[0, d]
                if union:
                    self.actions_.append(np.array([-1, 1, 0]))
                elif (xs[:, d] == x_d).all():
                    if (self.feature_constraints_[d] == "INC" and x_d == 1) or (
                        self.feature_constraints_[d] == "DEC" and x_d == 0
                    ):
                        self.actions_.append(np.array([0]))
                    else:
                        self.actions_.append(np.array([1 - 2 * x_d, 0]))
                else:
                    self.actions_.append(np.array([0]))
            else:
                x_min = np.max(xs[:, d]) if union else np.min(xs[:, d])
                x_max = np.min(xs[:, d]) if union else np.max(xs[:, d])
                if self.feature_constraints_[d] == "INC":
                    start = self.steps_[d]
                    stop = self.X_ub_[d] + self.steps_[d] - x_max
                elif self.feature_constraints_[d] == "DEC":
                    start = self.X_lb_[d] - x_min
                    stop = 0
                else:
                    start = self.X_lb_[d] - x_min
                    stop = self.X_ub_[d] + self.steps_[d] - x_max
                A_d = np.arange(start, stop, self.steps_[d])
                A_d = np.extract(abs(A_d) > self.tol_, A_d)
                if len(A_d) > self.max_candidates:
                    A_d = A_d[np.linspace(0, len(A_d) - 1, self.max_candidates, dtype=int)]
                A_d = np.append(A_d, 0)
                self.actions_.append(A_d)
        return self

    def setMultiCostSet(self, xs, y, cost_type="TLPS", p=1):
        self.costs_ = []
        for x in xs:
            cost_x = []
            if cost_type == "TLPS" or cost_type == "MPS":
                if self.Q_ == None:
                    self.Q_ = [
                        None
                        if self.feature_constraints_[d] == "FIX"
                        else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d])
                        for d in range(self.D_)
                    ]
                for d in range(self.D_):
                    if self.Q_[d] == None:
                        cost_x.append([0])
                    else:
                        Q_d = self.Q_[d]
                        Q_0 = Q_d(x[d])
                        cost_x.append(
                            [
                                abs(np.log2((1 - Q_d(x[d] + a)) / (1 - Q_0)))
                                if cost_type == "TLPS"
                                else abs(Q_d(x[d] + a) - Q_0)
                                for a in self.actions_[d]
                            ]
                        )
            elif cost_type == "SCM" or cost_type == "DACE":
                if cost_type == "SCM":
                    B_, _ = interaction_matrix(self.X_, interaction_type="causal")
                    B = np.eye(self.D_) - B_
                    C = self.getFeatureWeight(cost_type="standard")
                else:
                    self.cov_, B = interaction_matrix(
                        self.X_[self.Y_ == y] if len(self.Y_) == self.N_ else self.X_,
                        interaction_type="covariance",
                    )
                    C = self.getFeatureWeight(cost_type="uniform")
                for d in range(self.D_):
                    cost_d = []
                    for d_ in range(self.D_):
                        cost_d.append([C[d] * B[d][d_] * a for a in self.actions_[d_]])
                    cost_x.append(cost_d)
            else:
                weights = self.getFeatureWeight(cost_type=cost_type)
                if cost_type == "PCC":
                    p = 2
                for d in range(self.D_):
                    cost_x.append(list(weights[d] * abs(self.actions_[d]) ** p))
            self.costs_.append(cost_x)
        return self

    def generateActions(self, x, y, cost_type="TLPS", p=1, multi=False, union=False):
        if multi:
            self = self.setMultiActionSet(x, union=union)
            self = self.setMultiCostSet(x, y, cost_type=cost_type, p=p)
        else:
            self = self.setActionAndCost(x, y, cost_type=cost_type, p=p)
        return self.actions_, self.costs_

    def generateLOFParams(self, y, k=10, p=2, subsample=20, kernel="rbf"):
        lof = LocalOutlierFactor(
            n_neighbors=k,
            metric="manhattan" if p == 1 else "sqeuclidean",
            novelty=False,
        )
        X_lof = self.X_[self.Y_ == y]
        lof = lof.fit(X_lof)

        def k_distance(prototypes):
            return lof._distances_fit_X_[prototypes, k - 1]

        def local_reachability_density(prototypes):
            return lof._lrd[prototypes]

        prototypes = prototype_selection(X_lof, subsample=subsample, kernel=kernel)
        return (
            X_lof[prototypes],
            k_distance(prototypes),
            local_reachability_density(prototypes),
        )

    def mahalanobis_dist(self, x_1, x_2, y):
        if self.cov_ is None:
            self.cov_, _ = interaction_matrix(
                self.X_[self.Y_ == y] if len(self.Y_) == self.N_ else self.X_,
                interaction_type="covariance",
            )
        return mahalanobis(x_1, x_2, np.linalg.inv(self.cov_))

    def local_outlier_factor(self, x, y, k=10, p=2):
        lof = LocalOutlierFactor(
            n_neighbors=k, metric="manhattan" if p == 1 else "sqeuclidean", novelty=True
        )
        lof = lof.fit(self.X_[self.Y_ == y])
        return -lof.score_samples(x.reshape(1, -1))[0]

    def is_feasible(self, x_d, d):
        if self.feature_types_[d] == "B":
            return int(x_d) in [0, 1]
        else:
            return x_d >= self.X_lb_[d] and x_d <= self.X_ub_[d]

    def check_action_infeasible(self, x, a):
        x_new = x + a
        for d in range(self.D_):
            if not self.is_feasible(x_new[d], d):
                return True
        return False

    def cost(self, x, a, cost_type="TLPS"):
        cost = 0.0
        if cost_type == "TLPS" or cost_type == "MPS":
            if self.Q_ == None:
                self.Q_ = [
                    None
                    if self.feature_constraints_[d] == "FIX"
                    else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d])
                    for d in range(self.D_)
                ]
            for d in range(self.D_):
                if self.Q_[d] != None:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    if cost_type == "TLPS":
                        cost += abs(np.log2((1 - Q_d(x[d] + a[d])) / (1 - Q_0)))
                    else:
                        c = abs(Q_d(x[d] + a[d]) - Q_0)
                        if cost < c:
                            cost = c
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            if cost_type == "PCC":
                p = 2
            else:
                p = 1
            for d in range(self.D_):
                cost += weights[d] * (abs(a[d]) ** p)
        return cost


# class ActionCandidates


class ForestActionCandidates:
    def __init__(
        self,
        X,
        forest,
        Y=[],
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        feature_constraints=[],
        max_candidates=50,
        tol=1e-6,
    ):
        self.X_ = X
        self.Y_ = Y
        self.N_, self.D_ = X.shape
        self.feature_names_ = (
            feature_names
            if len(feature_names) == self.D_
            else ["x_{}".format(d) for d in range(self.D_)]
        )
        self.feature_types_ = (
            feature_types if len(feature_types) == self.D_ else ["C" for d in range(self.D_)]
        )
        self.feature_categories_ = feature_categories
        self.feature_constraints_ = (
            feature_constraints
            if len(feature_constraints) == self.D_
            else ["" for d in range(self.D_)]
        )
        self.tol_ = tol

        self.forest_ = forest
        self.T_ = forest.n_estimators
        self.trees_ = [t.tree_ for t in forest.estimators_]
        self.leaves_ = [np.where(tree.feature == -2)[0] for tree in self.trees_]
        self.L_ = [len(l) for l in self.leaves_]

        self.H_ = self.getForestLabels()
        self.ancestors_, self.regions_ = self.getForestRegions()
        self.thresholds_ = self.getForestThresholds()
        # self.M_ = [len(self.thresholds_[d])+1 for d in range(self.D_)]
        # self.partitions_ = self.getForestPartitions()

        self.X_lb_, self.X_ub_ = X.min(axis=0), X.max(axis=0)
        self.max_candidates = max_candidates
        self.steps_ = [
            (self.X_ub_[d] - self.X_lb_[d]) / max_candidates if self.feature_types_[d] == "C" else 1
            for d in range(self.D_)
        ]
        self.grids_ = [
            np.arange(self.X_lb_[d], self.X_ub_[d] + self.steps_[d], self.steps_[d])
            for d in range(self.D_)
        ]

        self.x_ = None
        self.actions_ = None
        self.costs_ = None
        self.Q_ = None
        self.cov_ = None
        self.I_ = None

    def getFeatureWeight(self, cost_type="uniform"):
        weights = np.ones(self.D_)
        if cost_type == "MAD":
            for d in range(self.D_):
                weight = mad(self.X_[:, d], scale="normal")
                if self.feature_types_[d] == "B" or abs(weight) < self.tol_:
                    weights[d] = (self.X_[:, d] * 1.4826).std()
                else:
                    weights[d] = weight**-1
        elif cost_type == "PCC" and len(self.Y_) == self.N_:
            for d in range(self.D_):
                weights[d] = abs(np.corrcoef(self.X_[:, d], self.Y_)[0, 1])
        elif cost_type == "standard":
            weights = np.std(self.X_, axis=0) ** -1
        elif cost_type == "normalize":
            weights = (self.X_.max(axis=0) - self.X_.min(axis=0)) ** -1
        elif cost_type == "robust":
            q25, q75 = np.percentile(self.X_, [0.25, 0.75], axis=0)
            for d in range(self.D_):
                if q75[d] - q25[d] == 0:
                    weights[d] = self.tol_**-1
                else:
                    weights = (q75[d] - q25) ** -1
        return weights

    def setActionSet(self, x, use_threshold=True):
        if (x == self.x_).all():
            return self
        self.x_ = x
        self.actions_ = []
        for d in range(self.D_):
            if self.feature_constraints_[d] == "FIX" or self.steps_[d] < self.tol_:
                self.actions_.append(np.array([0]))
            elif self.feature_types_[d] == "B":
                if (self.feature_constraints_[d] == "INC" and x[d] == 1) or (
                    self.feature_constraints_[d] == "DEC" and x[d] == 0
                ):
                    self.actions_.append(np.array([0]))
                else:
                    self.actions_.append(np.array([1 - 2 * x[d], 0]))
            else:
                if use_threshold:
                    A_d = (
                        self.thresholds_[d].astype(int) - x[d]
                        if self.feature_types_[d] == "I"
                        else self.thresholds_[d] - x[d]
                    )
                    A_d[A_d >= 0] += self.tol_ if self.feature_types_[d] == "C" else 1
                    if 0 not in A_d:
                        A_d = np.append(A_d, 0)
                    if self.feature_constraints_[d] == "INC":
                        A_d = np.extract(A_d >= 0, A_d)
                    elif self.feature_constraints_[d] == "DEC":
                        A_d = np.extract(A_d <= 0, A_d)
                else:
                    if self.feature_constraints_[d] == "INC":
                        start = x[d] + self.steps_[d]
                        stop = self.X_ub_[d] + self.steps_[d]
                    elif self.feature_constraints_[d] == "DEC":
                        start = self.X_lb_[d]
                        stop = x[d]
                    else:
                        start = self.X_lb_[d]
                        stop = self.X_ub_[d] + self.steps_[d]
                    A_d = np.arange(start, stop, self.steps_[d]) - x[d]
                    A_d = np.extract(abs(A_d) > self.tol_, A_d)
                    if len(A_d) > self.max_candidates:
                        A_d = A_d[np.linspace(0, len(A_d) - 1, self.max_candidates, dtype=int)]
                    A_d = np.append(A_d, 0)
                self.actions_.append(A_d)
        self = self.setForestIntervals(x)
        return self

    def setActionAndCost(self, x, y, cost_type="TLPS", p=1, use_threshold=True):
        self.costs_ = []
        self = self.setActionSet(x, use_threshold=use_threshold)
        if cost_type == "TLPS" or cost_type == "MPS":
            if self.Q_ == None:
                self.Q_ = [
                    None
                    if self.feature_constraints_[d] == "FIX"
                    else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d])
                    for d in range(self.D_)
                ]
            for d in range(self.D_):
                if self.Q_[d] == None:
                    self.costs_.append([0])
                else:
                    Q_d = self.Q_[d]
                    Q_0 = Q_d(x[d])
                    self.costs_.append(
                        [
                            abs(np.log2((1 - Q_d(x[d] + a)) / (1 - Q_0)))
                            if cost_type == "TLPS"
                            else abs(Q_d(x[d] + a) - Q_0)
                            for a in self.actions_[d]
                        ]
                    )
        elif cost_type == "SCM" or cost_type == "DACE":
            if cost_type == "SCM":
                B_, _ = interaction_matrix(self.X_, interaction_type="causal")
                B = np.eye(self.D_) - B_
                C = self.getFeatureWeight(cost_type="standard")
            else:
                self.cov_, B = interaction_matrix(
                    self.X_[self.Y_ == y] if len(self.Y_) == self.N_ else self.X_,
                    interaction_type="covariance",
                )
                C = self.getFeatureWeight(cost_type="uniform")
            for d in range(self.D_):
                cost_d = []
                for d_ in range(self.D_):
                    cost_d.append([C[d] * B[d][d_] * a for a in self.actions_[d_]])
                self.costs_.append(cost_d)
        else:
            weights = self.getFeatureWeight(cost_type=cost_type)
            for d in range(self.D_):
                self.costs_.append(list(weights[d] * abs(self.actions_[d]) ** p))
        return self

    def setMultiActionSet(self, xs, union=False, use_threshold=True):
        self.actions_ = []
        for d in range(self.D_):
            if self.feature_constraints_[d] == "FIX" or self.steps_[d] < self.tol_:
                self.actions_.append(np.array([0]))
            elif self.feature_types_[d] == "B":
                x_d = xs[0, d]
                if union:
                    self.actions_.append(np.array([-1, 1, 0]))
                elif (xs[:, d] == x_d).all():
                    if (self.feature_constraints_[d] == "INC" and x_d == 1) or (
                        self.feature_constraints_[d] == "DEC" and x_d == 0
                    ):
                        self.actions_.append(np.array([0]))
                    else:
                        self.actions_.append(np.array([1 - 2 * x_d, 0]))
                else:
                    self.actions_.append(np.array([0]))
            else:
                x_min = np.max(xs[:, d]) if union else np.min(xs[:, d])
                x_max = np.min(xs[:, d]) if union else np.max(xs[:, d])
                if use_threshold:
                    A_d = np.array([])
                    for x in xs:
                        A_d = np.concatenate(
                            [
                                A_d,
                                self.thresholds_[d].astype(int) - x[d]
                                if self.feature_types_[d] == "I"
                                else self.thresholds_[d] - x[d],
                            ]
                        )
                    A_d[A_d > 0] += self.tol_ if self.feature_types_[d] == "C" else 1
                    A_d = np.unique(A_d)
                    if self.feature_constraints_[d] == "INC":
                        A_d = np.extract(A_d >= 0, A_d)
                    elif self.feature_constraints_[d] == "DEC":
                        A_d = np.extract(A_d <= 0, A_d)
                    A_d = np.extract(x_min + A_d >= self.X_lb_[d], A_d)
                    A_d = np.extract(x_max + A_d <= self.X_ub_[d], A_d)
                    if A_d.shape[0] > self.max_candidates:
                        A_d = A_d[
                            np.linspace(
                                0,
                                A_d.shape[0],
                                self.max_candidates,
                                endpoint=False,
                                dtype=int,
                            )
                        ]
                    if 0 not in A_d:
                        A_d = np.append(A_d, 0)
                else:
                    if self.feature_constraints_[d] == "INC":
                        start = self.steps_[d]
                        stop = self.X_ub_[d] + self.steps_[d] - x_max
                    elif self.feature_constraints_[d] == "DEC":
                        start = self.X_lb_[d] - x_min
                        stop = 0
                    else:
                        start = self.X_lb_[d] - x_min
                        stop = self.X_ub_[d] + self.steps_[d] - x_max
                    A_d = np.arange(start, stop, self.steps_[d])
                    A_d = np.extract(abs(A_d) > self.tol_, A_d)
                    if len(A_d) > self.max_candidates:
                        A_d = A_d[np.linspace(0, len(A_d) - 1, self.max_candidates, dtype=int)]
                    A_d = np.append(A_d, 0)
                self.actions_.append(A_d)
        return self

    def setMultiCostSet(self, xs, y, cost_type="TLPS", p=1):
        self.costs_ = []
        for x in xs:
            cost_x = []
            if cost_type == "TLPS" or cost_type == "MPS":
                if self.Q_ == None:
                    self.Q_ = [
                        None
                        if self.feature_constraints_[d] == "FIX"
                        else CumulativeDistributionFunction(self.grids_[d], self.X_[:, d])
                        for d in range(self.D_)
                    ]
                for d in range(self.D_):
                    if self.Q_[d] == None:
                        cost_x.append([0])
                    else:
                        Q_d = self.Q_[d]
                        Q_0 = Q_d(x[d])
                        cost_x.append(
                            [
                                abs(np.log2((1 - Q_d(x[d] + a)) / (1 - Q_0)))
                                if cost_type == "TLPS"
                                else abs(Q_d(x[d] + a) - Q_0)
                                for a in self.actions_[d]
                            ]
                        )
            elif cost_type == "SCM" or cost_type == "DACE":
                if cost_type == "SCM":
                    B_, _ = interaction_matrix(self.X_, interaction_type="causal")
                    B = np.eye(self.D_) - B_
                    C = self.getFeatureWeight(cost_type="standard")
                else:
                    self.cov_, B = interaction_matrix(
                        self.X_[self.Y_ == y] if len(self.Y_) == self.N_ else self.X_,
                        interaction_type="covariance",
                    )
                    C = self.getFeatureWeight(cost_type="uniform")
                for d in range(self.D_):
                    cost_d = []
                    for d_ in range(self.D_):
                        cost_d.append([C[d] * B[d][d_] * a for a in self.actions_[d_]])
                    cost_x.append(cost_d)
            else:
                weights = self.getFeatureWeight(cost_type=cost_type)
                if cost_type == "PCC":
                    p = 2
                for d in range(self.D_):
                    cost_x.append(list(weights[d] * abs(self.actions_[d]) ** p))
            self.costs_.append(cost_x)
        return self

    def generateActions(
        self, x, y, cost_type="TLPS", p=1, use_threshold=True, multi=False, union=False
    ):
        if multi:
            self = self.setMultiActionSet(x, union=union, use_threshold=use_threshold)
            self = self.setMultiCostSet(x, y, cost_type=cost_type, p=p)
            self = self.setMultiForestIntervals(x)
        else:
            self = self.setActionAndCost(
                x, y, cost_type=cost_type, p=p, use_threshold=use_threshold
            )
        # return self.actions_, self.costs_, self.I_
        return self.actions_, self.costs_

    def setMultiForestIntervals(self, xs):
        self.I_ = []
        for x in xs:
            I_x = []
            for t in range(self.T_):
                I_t = []
                for l in range(self.L_[t]):
                    I_t_l = []
                    for d in range(self.D_):
                        xa = x[d] + self.actions_[d]
                        I_t_l.append(
                            list(
                                (
                                    (xa > self.regions_[t][l][d][0])
                                    & (xa <= self.regions_[t][l][d][1])
                                ).astype(int)
                            )
                        )
                    I_t.append(I_t_l)
                I_x.append(I_t)
            self.I_.append(I_x)
        return self

    def setForestIntervals(self, x):
        Is = [np.arange(len(a)) for a in self.actions_]
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    xa = x[d] + self.actions_[d]
                    I_t_l.append(
                        list(
                            (
                                (xa > self.regions_[t][l][d][0]) & (xa <= self.regions_[t][l][d][1])
                            ).astype(int)
                        )
                    )
                I_t.append(I_t_l)
            I.append(I_t)
        self.I_ = I
        return self

    #
    def getForestLabels(self):
        H = []
        for tree, leaves, l_t in zip(self.trees_, self.leaves_, self.L_):
            h_t = []
            stack = [0]
            while len(stack) != 0:
                i = stack.pop()
                if i in leaves:
                    val = tree.value[i][0]
                    h_t += [val[0] if val.shape[0] == 1 else val[1] / (val[0] + val[1])]
                else:
                    stack += [tree.children_right[i]]
                    stack += [tree.children_left[i]]
            H.append(h_t)
        return H

    def getForestRegions(self):
        As, Rs = [], []
        for tree, leaves in zip(self.trees_, self.leaves_):
            A, R = [], []
            stack = [[]]
            L, U = [[-np.inf] * self.D_], [[np.inf] * self.D_]
            node_stack = [0]
            while len(node_stack) != 0:
                n = node_stack.pop()
                a, l, u = stack.pop(), L.pop(), U.pop()
                if n in leaves:
                    A.append(a)
                    R.append([(l[d], u[d]) for d in range(self.D_)])
                else:
                    d = tree.feature[n]
                    if d not in a:
                        a_ = list(a) + [d]
                    stack.append(a_)
                    stack.append(a_)
                    # b = int(tree.threshold[n]) if self.feature_types_[d]=='I' else tree.threshold[n]
                    b = tree.threshold[n]
                    l_ = list(l)
                    u_ = list(u)
                    l[d] = b
                    u[d] = b
                    U.append(u_)
                    L.append(l)
                    node_stack.append(tree.children_right[n])
                    U.append(u)
                    L.append(l_)
                    node_stack.append(tree.children_left[n])
            As.append(A)
            Rs.append(R)
        return As, Rs

    def getForestThresholds(self):
        B = []
        for d in range(self.D_):
            b_d = []
            for tree in self.trees_:
                b_d += list(tree.threshold[tree.feature == d])
            b_d = list(set(b_d))
            b_d.sort()
            B.append(np.array(b_d))
        return B

    def getForestPartitions(self):
        I = []
        for t in range(self.T_):
            I_t = []
            for l in range(self.L_[t]):
                I_t_l = []
                for d in range(self.D_):
                    if self.regions_[t][l][d][0] == -np.inf:
                        start = 0
                    else:
                        start = self.thresholds_[d].index(self.regions_[t][l][d][0]) + 1
                    if self.regions_[t][l][d][1] == np.inf:
                        end = self.M_[d]
                    else:
                        end = self.thresholds_[d].index(self.regions_[t][l][d][1]) + 1
                    tmp = list(range(start, end))
                    I_t_l.append(tmp)
                I_t.append(I_t_l)
            I.append(I_t)
        return I

    def generateLOFParams(self, y, k=10, p=2, subsample=20, kernel="rbf"):
        lof = LocalOutlierFactor(
            n_neighbors=k,
            metric="manhattan" if p == 1 else "sqeuclidean",
            novelty=False,
        )
        X_lof = self.X_[self.Y_ == y]
        lof = lof.fit(X_lof)

        def k_distance(prototypes):
            return lof._distances_fit_X_[prototypes, k - 1]

        def local_reachability_density(prototypes):
            return lof._lrd[prototypes]

        prototypes = prototype_selection(X_lof, subsample=subsample, kernel=kernel)
        return (
            X_lof[prototypes],
            k_distance(prototypes),
            local_reachability_density(prototypes),
        )

    def mahalanobis_dist(self, x_1, x_2, y):
        if self.cov_ is None:
            self.cov_, _ = interaction_matrix(
                self.X_[self.Y_ == y] if len(self.Y_) == self.N_ else self.X_,
                interaction_type="covariance",
            )
        return mahalanobis(x_1, x_2, np.linalg.inv(self.cov_))

    def local_outlier_factor(self, x, y, k=10, p=2):
        lof = LocalOutlierFactor(
            n_neighbors=k, metric="manhattan" if p == 1 else "sqeuclidean", novelty=True
        )
        lof = lof.fit(self.X_[self.Y_ == y])
        return -lof.score_samples(x.reshape(1, -1))[0]


# class ForestActionCandidates


DATASETS = ["g", "w", "h", "c", "a", "d"]
DATASETS_NAME = {
    "g": "german",
    "w": "wine",
    "h": "fico",
    "c": "compas",
    "a": "adult",
    "d": "diabetes",
    "n": "nhanesi",
    "s": "student",
    "b": "bank",
    "i": "attrition",
    "t": "attrition",
}
DATASETS_FULLNAME = {
    "g": "German",
    "w": "WineQuality",
    "h": "FICO",
    "c": "COMPAS",
    "a": "Adult",
    "d": "Diabetes",
    "n": "NHANESI",
    "s": "StudentPerformance",
    "b": "BankMarketing",
    "i": "EmployeeAttrition",
    "t": "EmployeeAttrition",
}
DATASETS_PATH = {
    "g": "data/german.csv",
    "w": "data/wine.csv",
    "h": "data/heloc.csv",
    "c": "data/compas.csv",
    "a": "data/adult.csv",
    "d": "data/diabetes.csv",
    "n": "data/NHANESI.csv",
    "s": "data/student.csv",
    "b": "data/bank.csv",
    "i": "data/attrition.csv",
    "t": "data/toy_attrition.csv",
}
TARGET_NAME = {
    "g": "GoodCustomer",
    "w": "Quality",
    "h": "RiskPerformance",
    "c": "RecidivateWithinTwoYears",
    "a": "Income",
    "d": "Outcome",
    "n": "Mortality",
    "s": "Grade",
    "b": "subscribe",
    "i": "Attrition",
    "t": "Attrition",
}
TARGET_LABELS = {
    "g": ["Good", "Bad"],
    "w": [">5", "<=5"],
    "h": ["Good", "Bad"],
    "c": ["NotRecidivate", "Recidivate"],
    "a": [">50K", "<=50K"],
    "d": ["Good", "Bad"],
    "n": ["Surviving", "NotSurviving"],
    "s": [">10", "<=10"],
    "b": ["Yes", "No"],
    "i": ["No", "Yes"],
    "t": ["No", "Yes"],
}
FEATURE_NUMS = {
    "g": 40,
    "w": 12,
    "h": 23,
    "c": 14,
    "a": 108,
    "d": 8,
    "n": 17,
    "s": 48,
    "b": 35,
    "i": 44,
    "t": 27,
}
FEATURE_TYPES = {
    "g": ["I"] * 6 + ["B"] * 34,
    "w": ["C"] * 5 + ["I"] * 2 + ["C"] * 4 + ["B"],
    "h": ["I"] * 23,
    "c": ["I"] * 5 + ["B"] * 9,
    "a": ["I"] * 6 + ["B"] * 102,
    "d": ["I"] + ["C"] * 6 + ["I"],
    "n": ["I"] * 3 + ["C"] + ["I"] * 4 + ["C"] * 2 + ["I"] * 2 + ["C"] * 3 + ["I"] + ["B"],
    "s": ["I"] * 6 + ["B"] * 8 + ["I"] * 7 + ["B"] * 27,
    "b": ["I"] * 6 + ["B"] * 29,
    "i": ["I"] * 5 + ["B"] + ["I"] * 5 + ["B", "I", "B"] + ["I"] * 9 + ["B"] * 21,
    "t": ["I"] * 5 + ["B", "I", "B"] + ["I"] * 7 + ["B"] * 12,
}
FEATURE_CATEGORIES = {
    "g": [
        [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        [28, 29, 30],
        [31, 32, 33],
        [34, 35, 36],
    ],
    "w": [],
    "h": [],
    "c": [[5, 6, 7, 8, 9, 10], [11, 12]],
    "a": [
        list(range(6, 15)),
        list(range(15, 31)),
        list(range(31, 38)),
        list(range(38, 53)),
        list(range(53, 59)),
    ],
    "d": [],
    "n": [],
    "s": [
        list(range(21, 23)),
        list(range(23, 25)),
        list(range(25, 27)),
        list(range(27, 29)),
        list(range(29, 31)),
        list(range(31, 36)),
        list(range(36, 41)),
        list(range(41, 45)),
        list(range(45, 48)),
    ],
    "b": [
        list(range(9, 21)),
        list(range(21, 24)),
        list(range(24, 28)),
        list(range(28, 31)),
        list(range(31, 35)),
    ],
    "i": [
        list(range(23, 26)),
        list(range(26, 32)),
        list(range(32, 41)),
        list(range(41, 44)),
    ],
    "t": [list(range(15, 18)), list(range(18, 27))],
}
FEATURE_CONSTRAINTS = {
    "g": ["FIX"] + [""] * 36 + ["FIX"] * 3,
    "w": [""] * 11 + ["FIX"],
    "h": [""] * 23,
    "c": ["INC"] + [""] * 4 + ["FIX"] * 6 + [""] * 2 + ["FIX"],
    "a": ["INC"] + [""] * 57 + ["FIX"] * 50,
    "d": ["INC"] + [""] * 6 + ["INC"],
    "n": ["INC"] + [""] * 15 + ["FIX"],
    "s": ["FIX"] * 4 + [""] + ["FIX"] + [""] * 14 + ["FIX"] * 28,
    "b": ["FIX"] * 2 + ["", "INC", "", "INC"] + ["FIX"] * 22 + [""] * 7,
    "i": ["FIX", "", "FIX", "FIX", "", "FIX"]
    + [""] * 4
    + ["FIX"]
    + [""] * 8
    + ["INC"] * 4
    + [""] * 3
    + ["FIX"] * 6
    + [""] * 9
    + ["FIX"] * 3,
    "t": ["FIX", "", "FIX", "FIX"] + [""] * 3 + ["FIX"] * 3 + [""] + ["FIX"] * 4 + [""] * 12,
}


class DatasetHelper:
    def __init__(self, dataset="h", feature_prefix_index=False):
        self.dataset_ = dataset
        self.df_ = pd.read_csv(DATASETS_PATH[dataset], dtype="float")
        self.df_ = self.df_.sample(frac=0.02)
        self.y = self.df_[TARGET_NAME[dataset]].values
        self.X = self.df_.drop([TARGET_NAME[dataset]], axis=1).values
        self.n_samples, self.n_features = self.X.shape
        self.feature_names = list(self.df_.drop([TARGET_NAME[dataset]], axis=1).columns)
        if feature_prefix_index:
            self.feature_names = [
                "[x_{}]".format(d) + feat for d, feat in enumerate(self.feature_names)
            ]
        self.feature_types = FEATURE_TYPES[dataset]
        self.feature_categories = FEATURE_CATEGORIES[dataset]
        self.feature_constraints = FEATURE_CONSTRAINTS[dataset]
        self.target_name = TARGET_NAME[dataset]
        self.target_labels = TARGET_LABELS[dataset]
        self.dataset_name = DATASETS_NAME[dataset]
        self.dataset_fullname = DATASETS_FULLNAME[dataset]

    def train_test_split(self, test_size=0.25):
        return train_test_split(self.X, self.y, test_size=test_size)

    def to_markdown(self):
        feature_categories_flatten = sum(self.feature_categories, [])
        s = "| | Feature | Type | Value | Mutable |\n"
        s += "| --- | --- | --- | :---: | :---: |\n"
        i = 1
        for d in range(self.n_features):
            if d not in feature_categories_flatten:
                if self.feature_types[d] == "C":
                    s += "| {} | {} | Continuous | [{:.4f}, {:.4f}] | {} |\n".format(
                        i,
                        self.feature_names[d],
                        int(self.X[:, d].min()),
                        int(self.X[:, d].max()),
                        self.feature_constraints[d] != "FIX",
                    )
                if self.feature_types[d] == "B":
                    s += "| {} | {} | Bool | {{False: 0, True: 1}} | {} |\n".format(
                        i, self.feature_names[d], self.feature_constraints[d] != "FIX"
                    )
                else:
                    s += "| {} | {} | Integer | [{}, {}] | {} |\n".format(
                        i,
                        self.feature_names[d],
                        int(self.X[:, d].min()),
                        int(self.X[:, d].max()),
                        self.feature_constraints[d] != "FIX",
                    )
                i += 1
        for G in self.feature_categories:
            prv, nxt = self.feature_names[G[0]].split(":")
            s += "| {} | {} | Category | {} ".format(i, prv, nxt)
            for d in G[1:]:
                _, nxt = self.feature_names[d].split(":")
                s += "<br> {} ".format(nxt)
            s += "| {} |\n".format(self.feature_constraints[d] != "FIX")
            i += 1
        return s


# class DatasetHelper


def synthetic_dataset(N=100, split=False, test_size=0.25):
    X_cat = np.zeros([N, 3])
    X_cat[: int(N / 2), 0] = 1
    X_cat[int(N / 2) : int((3 * N) / 4), 1] = 1
    X_cat[int((3 * N) / 4) :, 2] = 1
    X = np.concatenate(
        [np.random.randint(0, 100, [N, 1]), np.random.randint(0, 2, [N, 1]), X_cat],
        axis=1,
    )
    y = ((X[:, 0] < 50) + (X[:, 4] == 1)).astype(int)
    feature_names = ["Score", "Skilled", "Course:A", "Course:B", "Course:C"]
    feature_types = ["I", "B", "B", "B", "B"]
    feature_categories = [[2, 3, 4]]
    feature_constraints = ["INC", "", "", "", ""]
    target_name = "Success"
    target_labels = ["Yes", "No"]
    dataset_name = "synthetic"
    dataset_fullname = "Synthetic"
    if split:
        return (
            train_test_split(X, y, test_size=test_size),
            feature_names,
            feature_types,
            feature_categories,
            feature_constraints,
            target_name,
            target_labels,
            dataset_name,
            dataset_fullname,
        )
    else:
        return (
            X,
            y,
            feature_names,
            feature_types,
            feature_categories,
            feature_constraints,
            target_name,
            target_labels,
            dataset_name,
            dataset_fullname,
        )


def _check_action_and_cost(dataset="h"):
    DH = DatasetHelper(dataset=dataset)
    AC = ActionCandidates(
        DH.X[1:],
        feature_names=DH.feature_names,
        feature_types=DH.feature_types,
        feature_categories=DH.feature_categories,
        feature_constraints=DH.feature_constraints,
    )
    print(DH.X[0])
    A, C = AC.generateActions(DH.X[0], cost_type="TLPS")
    for d, a in zip(AC.feature_names_, A):
        print(d, a)
    for d, c in zip(AC.feature_names_, C):
        print(d, c)


def _check_forest_action(dataset="h"):
    from sklearn.ensemble import RandomForestClassifier

    DH = DatasetHelper(dataset=dataset)
    X, y = DH.X, DH.y
    forest = RandomForestClassifier(n_estimators=2, max_depth=4)
    forest = forest.fit(X[1:], y[1:])
    AC = ForestActionCandidates(
        DH.X[1:],
        forest,
        feature_names=DH.feature_names,
        feature_types=DH.feature_types,
        feature_categories=DH.feature_categories,
        feature_constraints=DH.feature_constraints,
    )
    (
        _,
        _,
        I,
    ) = AC.generateActions(X[0], cost_type="unifrom", use_threshold=True)
    for I_t in I:
        for I_t_l in I_t:
            print(I_t_l)
            print(flatten(I_t_l))


if __name__ == "__main__":
    # _check_action_and_cost(dataset='h')
    _check_forest_action(dataset="d")
