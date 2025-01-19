import numpy as np
from utils import flatten, Cost
from pyfpgrowth import find_frequent_patterns
from copy import deepcopy
import time


class FeatureDiscretizer:
    def __init__(self, bins=5, strategy="quantile", negation=False, onehot=False):
        self.bins = bins
        self.strategy = strategy
        self.negation = negation
        self.onehot = onehot

    def fit(self, X, feature_types=[], feature_names=[], feature_categories=[]):
        self.D = X.shape[1]
        self.feature_types = (
            feature_types
            if len(feature_types) == self.D
            else ["C" for d in range(self.D)]
        )
        self.feature_names = (
            feature_names
            if len(feature_names) == self.D
            else ["x_{}".format(d) for d in range(self.D)]
        )
        self.feature_categories_ = (
            feature_categories if len(feature_names) == self.D else []
        )
        self.feature_categories_flatten = flatten(feature_categories)

        self.edges = []
        self.discretized_feature_names = []
        self.discretized_feature_indices = []
        self.discretized_feature_dict = []
        i = 0
        for d in range(self.D):
            if self.feature_types[d] == "B":
                self.edges += [[]]
                if self.negation or d not in self.feature_categories_flatten:
                    self.discretized_feature_names += [
                        self.feature_names[d] + "=True",
                        self.feature_names[d] + "=False",
                    ]
                    self.discretized_feature_indices += [[i, i + 1]]
                    self.discretized_feature_dict += [d, d]
                    i += 2
                else:
                    self.discretized_feature_names += [self.feature_names[d]]
                    self.discretized_feature_indices += [[i]]
                    self.discretized_feature_dict += [d]
                    i += 1
            else:
                bins = self.bins + 2
                while True:
                    if self.strategy == "uniform":
                        edge_candidates = (
                            np.linspace(X[:, d].min(), X[:, d].max(), bins).astype(int)
                            if self.feature_types[d] == "I"
                            else np.linspace(X[:, d].min(), X[:, d].max(), bins)
                        )
                    elif self.strategy == "quantile":
                        edge_candidates = (
                            np.asarray(
                                np.percentile(X[:, d], np.linspace(0, 100, bins))
                            ).astype(int)
                            if self.feature_types[d] == "I"
                            else np.asarray(
                                np.percentile(X[:, d], np.linspace(0, 100, bins))
                            )
                        )
                    edges = np.unique(edge_candidates)[1:-1]
                    if len(edges) > 0:
                        break
                    else:
                        bins += 1
                self.edges += [edges]
                if self.onehot:
                    for j in range(edges.shape[0]):
                        if j == 0:
                            self.discretized_feature_names += [
                                self.feature_names[d] + "<{}".format(edges[j])
                                if self.feature_types[d] == "I"
                                else self.feature_names[d] + "<{:.4}".format(edges[j])
                            ]
                        else:
                            self.discretized_feature_names += [
                                "{}<={}<{}".format(
                                    edges[j - 1], self.feature_names[d], edges[j]
                                )
                                if self.feature_types[d] == "I"
                                else "{:.4}<={}<{:.4}".format(
                                    edges[j - 1], self.feature_names[d], edges[j]
                                )
                            ]
                    self.discretized_feature_names += [
                        "{}>={}".format(self.feature_names[d], edges[j])
                        if self.feature_types[d] == "I"
                        else "{}>={:.4}".format(self.feature_names[d], edges[j])
                    ]
                    actual_bins = edges.shape[0] + 1
                else:
                    for j, t in enumerate(edges):
                        self.discretized_feature_names += [
                            self.feature_names[d] + "<={}".format(t)
                            if self.feature_types[d] == "I"
                            else self.feature_names[d] + "<={:.4}".format(t)
                        ]
                        if self.negation:
                            self.discretized_feature_names += [
                                self.feature_names[d] + ">{}".format(t)
                                if self.feature_types[d] == "I"
                                else self.feature_names[d] + ">{:.4}".format(t)
                            ]
                    actual_bins = (
                        edges.shape[0] * 2 if self.negation else edges.shape[0]
                    )

                self.discretized_feature_indices += [list(range(i, i + actual_bins))]
                self.discretized_feature_dict += [d] * actual_bins
                i += actual_bins

        return self

    def transform(self, X):
        X_new_list = []
        for d in range(self.D):
            if self.feature_types[d] == "B":
                if self.negation or d not in self.feature_categories_flatten:
                    X_new_list += [np.array([X[:, d], 1 - X[:, d]]).T]
                else:
                    X_new_list += [X[:, [d]]]
            else:
                if self.onehot:
                    X_bin = np.zeros([X.shape[0], self.edges[d].shape[0] + 1])
                    for j in range(self.edges[d].shape[0]):
                        if j == 0:
                            X_bin[:, j] = (X[:, d] < self.edges[d][j]).astype(int)
                        else:
                            idx = np.where(
                                (X[:, d] >= self.edges[d][j - 1])
                                & (X[:, d] < self.edges[d][j])
                            )[0]
                            X_bin[idx, j] = 1
                    X_bin[:, -1] = (X[:, d] > self.edges[d][-1]).astype(int)
                else:
                    X_bin = np.zeros(
                        [
                            X.shape[0],
                            self.edges[d].shape[0] * 2
                            if self.negation
                            else self.edges[d].shape[0],
                        ]
                    )
                    for j, t in enumerate(self.edges[d]):
                        i = 2 * j if self.negation else j
                        X_bin[:, i] = (X[:, d] <= t).astype(int)
                        if self.negation:
                            X_bin[:, i + 1] = (X[:, d] > t).astype(int)
                X_new_list += [X_bin]
        return np.concatenate(X_new_list, axis=1)

    def discretization_summary(self):
        ret = []
        for d in range(self.D):
            if self.feature_types[d] == "B":
                if self.negation or d not in self.feature_categories_flatten:
                    ret += [
                        {"feature": d, "operator": "I", "threshold": (1, np.inf)},
                        {"feature": d, "operator": "I", "threshold": (-np.inf, 1)},
                    ]
                else:
                    ret += [{"feature": d, "operator": "E", "threshold": 1}]
            else:
                if self.onehot:
                    for j in range(self.edges[d].shape[0]):
                        if j == 0:
                            ret += [
                                {
                                    "feature": d,
                                    "operator": "I",
                                    "threshold": (-np.inf, self.edges[d][j]),
                                }
                            ]
                        else:
                            ret += [
                                {
                                    "feature": d,
                                    "operator": "I",
                                    "threshold": (
                                        self.edges[d][j - 1],
                                        self.edges[d][j],
                                    ),
                                }
                            ]
                    ret += [
                        {
                            "feature": d,
                            "operator": "I",
                            "threshold": (self.edges[d][j], np.inf),
                        }
                    ]
                else:
                    for j, t in enumerate(self.edges[d]):
                        if self.negation:
                            ret += [
                                {"feature": d, "operator": "L", "threshold": t},
                                {"feature": d, "operator": "G", "threshold": t},
                            ]
                        else:
                            ret += [{"feature": d, "operator": "L", "threshold": t}]
        return ret


class FrequentRuleMiner:
    def __init__(self, minsup=0.8, discretization=False, negation=False, onehot=False):
        self.minsup_ = minsup
        self.discretization_ = discretization
        self.negation_ = negation
        self.onehot_ = onehot

    def __str__(self):
        s = ""
        for l, rule in enumerate(self.rule_names_):  # noqa: E741
            s += "Rule {:3}: ".format(l)
            s += rule
            s += "\n"
        return s

    def __getSupp(self, X, rule):
        return X[:, rule].prod(axis=1).sum() + 1e-8

    def setRuleNames(self):
        self.rule_names_ = []
        for rule in self.rules_:
            buf = ""
            buf += "'" + self.feature_names[rule[0]] + "'"
            for r in rule[1:]:
                buf += " AND '" + self.feature_names[r] + "'"
            self.rule_names_.append(buf)
        return self

    # [0,1,0,1,...] -> [1,3,...]
    def OnehotsToTransactions(self, X):
        transaction = []
        for x in X:
            transaction.append(np.where(x == 1)[0])
        return transaction

    def miningFrequentRules(self, X, max_L=8):
        N = X.shape[0]
        threshold = self.minsup_ if self.minsup_ in range(1, N) else N * self.minsup_
        transaction = self.OnehotsToTransactions(X)
        patterns = find_frequent_patterns(transaction, threshold)
        return [rule for rule in list(patterns.keys()) if len(rule) <= max_L]

    def fit(
        self,
        X,
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        max_L=8,
        discretization_bins=10,
        discretization_strategy="quantile",
        save_file="",
    ):
        if self.discretization_:
            self.fd_ = FeatureDiscretizer(
                bins=discretization_bins,
                strategy=discretization_strategy,
                negation=self.negation_,
                onehot=self.onehot_,
            ).fit(
                X,
                feature_types=feature_types,
                feature_names=feature_names,
                feature_categories=feature_categories,
            )
            self.feature_names = self.fd_.discretized_feature_names
            X = self.fd_.transform(X)
        else:
            self.feature_names = feature_names

        self.D_ = X.shape[1]
        self.rules_ = self.miningFrequentRules(X, max_L=max_L)
        self.length_ = len(self.rules_)

        # rule indicator matrix (D * length)
        self.Z_ = np.zeros([self.length_, self.D_], dtype=int)
        for l, rule in enumerate(self.rules_):  # noqa: E741
            self.Z_[l, rule] = 1
        self.L_ = self.Z_.sum(axis=1)
        self = self.setRuleNames()
        if len(save_file) != 0:
            np.savetxt(save_file, self.Z_, delimiter=",", fmt="%d")
        return self

    def transform(self, X):
        if self.discretization_:
            X = self.fd_.transform(X)
        return (X.dot(self.Z_.T) / self.L_).astype(int)


class AReS:
    def __init__(
        self,
        mdl,
        X,
        Y=[],
        max_rule=8,
        max_rule_length=8,
        minimum_support=0.6,
        discretization_bins=10,
        max_candidates=50,
        use_probability=True,
        print_objective=True,
        tol=1e-6,
        feature_names=[],
        feature_types=[],
        feature_categories=[],
        feature_constraints=[],
        target_name="Output",
        target_labels=["Good", "Bad"],
    ):
        self.mdl_ = mdl
        self.cost_ = Cost(
            X,
            Y,
            feature_types=feature_types,
            feature_categories=feature_categories,
            feature_constraints=feature_constraints,
            max_candidates=max_candidates,
            tol=tol,
        )

        self.max_rule_ = max_rule
        self.max_rule_length_ = max_rule_length
        self.feature_names_ = (
            feature_names
            if len(feature_names) == X.shape[1]
            else ["x_{}".format(d) for d in range(X.shape[1])]
        )
        self.feature_types_ = (
            feature_types
            if len(feature_types) == X.shape[1]
            else ["C" for d in range(X.shape[1])]
        )
        self.feature_categories_ = feature_categories
        self.feature_categories_flatten_ = flatten(feature_categories)
        self.feature_constraints_ = (
            feature_constraints
            if len(feature_constraints) == X.shape[1]
            else ["" for d in range(X.shape[1])]
        )
        self.target_name_ = target_name
        self.target_labels_ = target_labels
        self.tol_ = tol
        self.feature_categories_inv_ = []
        for d in range(X.shape[1]):
            g = -1
            if self.feature_types_[d] == "B":
                for i, cat in enumerate(self.feature_categories_):
                    if d in cat:
                        g = i
                        break
            self.feature_categories_inv_.append(g)

        self.rule_miner_ = FrequentRuleMiner(
            minsup=minimum_support, discretization=True, negation=False, onehot=True
        ).fit(
            X,
            feature_names=feature_names,
            feature_types=feature_types,
            feature_categories=feature_categories,
            max_L=max_rule_length,
            discretization_bins=discretization_bins,
            discretization_strategy="quantile",
        )
        self.rule_names_ = self.rule_miner_.rule_names_
        self.rule_to_discretized_feature_ = self.rule_miner_.rules_
        self.discretized_feature_names_ = self.rule_miner_.feature_names
        self.discretized_feature_to_feature_ = (
            self.rule_miner_.fd_.discretized_feature_dict
        )
        self.feature_to_discretized_feature_ = (
            self.rule_miner_.fd_.discretized_feature_indices
        )
        self.discretized_feature_summary_ = (
            self.rule_miner_.fd_.discretization_summary()
        )
        self.P_ = len(self.rule_names_)
        self.use_probability_ = use_probability
        self.print_objective_ = print_objective

        # generate recourse-rule candidates
        # print('< Generate Recourse-Rules >')
        start = time.perf_counter()
        self.R_candidates_ = []
        for p in range(self.P_):
            if False:
                if (p + 1) % 100 == 0:
                    print(p + 1, "/", self.P_)
            d_p = [
                self.discretized_feature_to_feature_[dd]
                for dd in self.rule_to_discretized_feature_[p]
            ]
            summary_p = [
                self.discretized_feature_summary_[dd]
                for dd in self.rule_to_discretized_feature_[p]
            ]
            for q in [q for q in range(self.P_) if q != p]:
                d_q = [
                    self.discretized_feature_to_feature_[dd]
                    for dd in self.rule_to_discretized_feature_[q]
                ]
                summary_q = [
                    self.discretized_feature_summary_[dd]
                    for dd in self.rule_to_discretized_feature_[q]
                ]
                features = []
                values = []
                fail = False
                for j, d in enumerate(d_q):
                    if (
                        self.feature_types_[d] == "B"
                        and d in self.feature_categories_flatten_
                    ):
                        categories = [
                            d_
                            for d_ in d_p
                            if d_
                            in self.feature_categories_[self.feature_categories_inv_[d]]
                            and d_ != d
                        ]
                        if len(categories) < 1:
                            fail = True
                            break
                        if self.feature_constraints_[d] == "FIX":
                            fail = True
                            break
                        for d_ in categories:
                            if d < d_:
                                values += [1, 0]
                                features += [d, d_]
                            else:
                                values += [0, 1]
                                features += [d_, d]
                    else:
                        if d not in d_p:
                            fail = True
                            break
                        l_p, u_p = summary_p[d_p.index(d)]["threshold"]
                        l_q, u_q = summary_q[j]["threshold"]
                        if abs(l_p - l_q) < self.tol_ or abs(u_p - u_q) < self.tol_:
                            continue
                        if self.feature_constraints_[d] == "FIX":
                            fail = True
                            break
                        if u_p <= l_q:
                            if self.feature_constraints_[d] == "DEC":
                                fail = True
                                break
                            values += [l_q]
                            features += [d]
                        else:
                            if self.feature_constraints_[d] == "INC":
                                fail = True
                                break
                            values += [
                                u_q - 1
                                if self.feature_types_[d] == "I"
                                or self.feature_types_[d] == "B"
                                else u_q - self.tol_
                            ]
                            features += [d]
                if len(features) != 0 and not fail:
                    r = {}
                    r["antecedent"] = p
                    r["consequent"] = q
                    r["feature"] = np.array(features)
                    r["value"] = np.array(values)
                    self.R_candidates_ += [r]
        self.preprocess_time_ = time.perf_counter() - start
        self.feasible_ = False

    def is_satisfy(self, rule, X_rule):
        return X_rule[:, rule] == 1

    def cover(self, rule, X_rule):
        return np.where(X_rule[:, rule] == 1)[0]

    def substitute(self, X, X_rule, r):
        X_ = X.copy()
        for d, v in zip(r["feature"], r["value"]):
            X_[self.cover(r["antecedent"], X_rule), d] = v
        return X_

    def change_num(self, features):
        x = 0
        for d in features:
            if d in self.feature_categories_flatten_:
                x += 1 / 2
            else:
                x += 1
        return x

    def setProblem(self, X, max_change_num=4, cost_type="TLPS"):
        self.N_, self.D_ = X.shape
        self.max_change_num_ = max_change_num
        self.cost_type_ = cost_type
        X_rule = self.rule_miner_.transform(X)
        R_candidates = []
        for r in self.R_candidates_:
            r["cov"] = self.cover(r["antecedent"], X_rule)
            if (
                r["cov"].shape[0] > 0
                and self.change_num(r["feature"]) <= max_change_num
            ):
                X_sub = self.substitute(X, X_rule, r)
                r["acc"] = np.where(
                    (self.is_satisfy(r["antecedent"], X_rule))
                    & (self.mdl_.predict(self.substitute(X, X_rule, r)) == 0)
                )[0]
                r["inc"] = r["cov"].shape[0] - r["acc"].shape[0]
                r["cst"] = np.array(
                    [
                        self.cost_.compute(x, a, cost_type=cost_type)
                        for x, a in zip(X[r["cov"]], X_sub[r["cov"]] - X[r["cov"]])
                    ]
                ).mean()
                r["probability"] = r["cov"].shape[0] / self.N_
                R_candidates += [r]
        return R_candidates

    # greedy optimization: Obj. = lambda_acc * acc + lambda_cov * cov - lambda_cst * cst
    def greedySolve(
        self,
        R_candidates,
        lambda_acc=1.0,
        lambda_cov=1.0,
        lambda_cst=0.01,
        objective="arange",
    ):
        ## initial step
        if objective == "origin":
            i = np.argmax(
                [
                    -lambda_acc * r["inc"]
                    + lambda_cov * r["cov"].shape[0]
                    - lambda_cst * r["cst"]
                    for r in R_candidates
                ]
            )
        else:
            i = np.argmax(
                [
                    lambda_acc * r["acc"].shape[0]
                    + lambda_cov * r["cov"].shape[0]
                    - lambda_cst * r["cst"]
                    for r in R_candidates
                ]
            )
        r_initial = R_candidates.pop(i)
        acc_indices = r_initial["acc"]
        cov_indices = r_initial["cov"]
        cst_sums = r_initial["cst"]
        inc_sums = r_initial["inc"]
        R = [r_initial]

        ## optimization loop
        objs = {"cov": [], "acc": [], "cst": [], "inc": []}
        while len(R) < self.max_rule_:
            if objective == "origin":
                i = np.argmax(
                    [
                        -lambda_acc * (inc_sums + r["inc"])
                        + lambda_cov * np.union1d(cov_indices, r["cov"]).shape[0]
                        - lambda_cst * (cst_sums + r["cst"])
                        for r in R_candidates
                    ]
                )
            else:
                i = np.argmax(
                    [
                        lambda_acc * np.union1d(acc_indices, r["acc"]).shape[0]
                        + lambda_cov * np.union1d(cov_indices, r["cov"]).shape[0]
                        - lambda_cst * (cst_sums + r["cst"])
                        for r in R_candidates
                    ]
                )
            r_i = R_candidates.pop(i)
            R += [r_i]
            acc_indices = np.union1d(acc_indices, r_i["acc"])
            cov_indices = np.union1d(cov_indices, r_i["cov"])
            cst_sums += r_i["cst"]
            inc_sums += r_i["inc"]
            objs["cov"].append(cov_indices.shape[0])
            objs["acc"].append(acc_indices.shape[0])
            objs["cst"].append(cst_sums)
            objs["inc"].append(inc_sums)
        return R, acc_indices, cov_indices, cst_sums, inc_sums, objs

    def setDefaultRule(
        self,
        X,
        cov_indices,
        cost_type="TLPS",
        lambda_acc=1.0,
        lambda_cov=1.0,
        lambda_cst=0.01,
        objective="arange",
    ):
        default_rule = {}
        uncov = np.array([n for n in range(self.N_) if n not in cov_indices])
        if uncov.shape[0] > 0:
            default_rule["determined"] = True
            default_rule["cov"] = uncov
            obj_opt = -np.inf
            for p in range(self.P_):
                X_ = X[uncov].copy()
                d_p = [
                    self.discretized_feature_to_feature_[dd]
                    for dd in self.rule_to_discretized_feature_[p]
                ]
                t_p = [
                    self.discretized_feature_summary_[dd]["threshold"]
                    for dd in self.rule_to_discretized_feature_[p]
                ]
                value = []
                feature = []
                for i, d in enumerate(d_p):
                    if (
                        d in self.feature_categories_flatten_
                        or self.feature_constraints_[d] == "FIX"
                    ):
                        continue
                    l_d, u_d = t_p[i]
                    for x_ in X_:
                        if x_[d] < l_d and self.feature_constraints_[d] != "DEC":
                            x_[d] = l_d
                        elif x_[d] >= u_d and self.feature_constraints_[d] != "INC":
                            x_[d] = (
                                u_d - 1
                                if self.feature_types_[d] == "I"
                                or self.feature_types_[d] == "B"
                                else u_d - self.tol_
                            )
                    feature += [d]
                    value += [(l_d, u_d)]
                acc = uncov[self.mdl_.predict(X_) == 0]
                cst = np.array(
                    [
                        self.cost_.compute(x, x_ - x, cost_type=cost_type)
                        for x, x_ in zip(X[uncov], X_)
                    ]
                ).mean()
                inc = uncov.shape[0] - acc.shape[0]
                if objective == "origin":
                    obj = -lambda_acc * inc + lambda_cov * 1.0 - lambda_cst * cst
                else:
                    obj = (
                        lambda_acc * acc.shape[0] + lambda_cov * 1.0 - lambda_cst * cst
                    )
                if obj > obj_opt:
                    default_rule["feature"] = feature
                    default_rule["value"] = value
                    default_rule["cst"] = cst
                    default_rule["acc"] = acc
                    default_rule["inc"] = inc
                    default_rule["consequent"] = p
                    obj_opt = obj
        else:
            default_rule["determined"] = False
            default_rule["cov"] = np.array([])
            default_rule["feature"] = np.array([])
            default_rule["value"] = np.array([])
            default_rule["cst"] = 0.0
            default_rule["acc"] = 0.0
            default_rule["inc"] = 0.0
            default_rule["consequent"] = None
        return default_rule

    def fit(
        self,
        X,
        max_change_num=4,
        cost_type="TLPS",
        lambda_acc=1.0,
        lambda_cov=1.0,
        lambda_cst=0.01,
        objective="origin",
        verbose=True,
    ):
        self.lambda_acc = lambda_acc
        self.lambda_cov = lambda_cov
        self.lambda_cst = lambda_cst
        start = time.perf_counter()

        R_candidates = self.setProblem(
            X, max_change_num=max_change_num, cost_type=cost_type
        )
        if len(R_candidates) == 0:
            print("No candidate recourse-rule.")
            self.R_ = []
            self.default_rule_ = {"determined": False}
            return self

        R, acc_indices, cov_indices, cst_sums, inc_sums, objs = self.greedySolve(
            R_candidates,
            lambda_acc=lambda_acc,
            lambda_cov=lambda_cov,
            lambda_cst=lambda_cst,
            objective=objective,
        )
        self.acc_ = acc_indices.shape[0]
        self.cov_ = cov_indices.shape[0]
        self.cst_ = cst_sums
        self.inc_ = inc_sums
        if objective == "origin":
            self.obj_ = (
                -lambda_acc * self.inc_
                + lambda_cov * self.cov_
                - lambda_cst * self.cst_
            )
        else:
            self.obj_ = (
                lambda_acc * self.acc_ + lambda_cov * self.cov_ - lambda_cst * self.cst_
            )
        self.R_ = R
        self.objs_ = objs
        self.feasible_ = True
        self.default_rule_ = self.setDefaultRule(
            X,
            cov_indices,
            cost_type=cost_type,
            lambda_acc=lambda_acc,
            lambda_cov=lambda_cov,
            lambda_cst=lambda_cst,
            objective=objective,
        )

        self.time_ = time.perf_counter() - start
        self.objective_ = objective
        return self

    def tuning(
        self,
        X,
        X_vl=None,
        max_change_num=4,
        cost_type="TLPS",
        gamma=1.0,
        lambda_acc=[1.0],
        lambda_cov=[1.0],
        lambda_cst=[1.0],
        objective="origin",
        after_fit=False,
        verbose=True,
    ):
        if X_vl is None:
            X_vl = X
        start = time.perf_counter()

        R_candidates = self.setProblem(
            X, max_change_num=max_change_num, cost_type=cost_type
        )
        if len(R_candidates) == 0:
            print("No candidate recourse-rule.")
            return [0.0, 0.0, 0.0]

        obj_best = np.inf
        lambda_best = [0.0, 0.0, 0.0]
        start = time.perf_counter()
        for l_acc in lambda_acc:
            for l_cov in lambda_cov:
                for l_cst in lambda_cst:
                    R, acc_indices, cov_indices, cst_sums, inc_sums, objs = (
                        self.greedySolve(
                            deepcopy(R_candidates),
                            lambda_acc=l_acc,
                            lambda_cov=l_cov,
                            lambda_cst=l_cst,
                            objective=objective,
                        )
                    )
                    self.cov_ = cov_indices.shape[0]
                    self.acc_ = acc_indices.shape[0]
                    self.cst_ = cst_sums
                    self.inc_ = inc_sums
                    if objective == "origin":
                        self.obj_ = (
                            -l_acc * self.inc_ + l_cov * self.cov_ - l_cst * self.cst_
                        )
                    else:
                        self.obj_ = (
                            l_acc * self.acc_ + l_cov * self.cov_ - l_cst * self.cst_
                        )
                    self.R_ = R
                    self.objs_ = objs
                    self.feasible_ = True
                    self.default_rule_ = self.setDefaultRule(
                        X,
                        cov_indices,
                        cost_type=cost_type,
                        lambda_acc=l_acc,
                        lambda_cov=l_cov,
                        lambda_cst=l_cst,
                        objective=objective,
                    )
                    obj = self.cost(X_vl, cost_type=cost_type) + gamma * self.loss(X_vl)
                    if True:
                        print(
                            "- (l_acc, l_cov, l_cst, obj.) = ({}, {}, {}, {}): obj. = {} ({} [s])".format(
                                l_acc,
                                l_cov,
                                l_cst,
                                self.obj_,
                                obj,
                                time.perf_counter() - start,
                            )
                        )
                    if obj < obj_best:
                        obj_best = obj
                        lambda_best = [l_acc, l_cov, l_cst]
        if True:
            print(
                "- *Best* (l_acc, l_cov, l_cst) = ({}, {}, {}): obj. = {}".format(
                    lambda_best[0], lambda_best[1], lambda_best[2], obj_best
                )
            )
        self.objective_ = objective
        if after_fit:
            l_acc, l_cov, l_cst = lambda_best
            self.lambda_acc, self.lambda_cov, self.lambda_cst = lambda_best
            R, acc_indices, cov_indices, cst_sums, inc_sums, objs = self.greedySolve(
                R_candidates,
                lambda_acc=l_acc,
                lambda_cov=l_cov,
                lambda_cst=l_cst,
                objective=objective,
            )
            self.acc_ = acc_indices.shape[0]
            self.cov_ = cov_indices.shape[0]
            self.cst_ = cst_sums
            self.inc_ = inc_sums
            if objective == "origin":
                self.obj_ = -l_acc * self.inc_ + l_cov * self.cov_ - l_cst * self.cst_
            else:
                self.obj_ = l_acc * self.acc_ + l_cov * self.cov_ - l_cst * self.cst_
            self.R_ = R
            self.objs_ = objs
            self.feasible_ = True
            self.default_rule_ = self.setDefaultRule(
                X,
                cov_indices,
                cost_type=cost_type,
                lambda_acc=l_acc,
                lambda_cov=l_cov,
                lambda_cst=l_cst,
                objective=objective,
            )
            self.time_ = time.perf_counter() - start
            return self
        else:
            return lambda_best

    def __str__(self):
        s = ""
        for r in self.R_:
            s += "- If {} (Cov. = {}/{} = {:.1%}):\n".format(
                self.rule_names_[r["antecedent"]],
                r["cov"].shape[0],
                self.N_,
                r["probability"],
            )
            s += "\t* Recourse Rule (Acc. = {}/{} = {:.1%}, Cost = {:.4})\n".format(
                r["acc"].shape[0],
                r["cov"].shape[0],
                r["acc"].shape[0] / r["cov"].shape[0],
                r["cst"],
            )
            for dd in self.rule_to_discretized_feature_[r["consequent"]]:
                s += "\t\t* {}\n".format(self.discretized_feature_names_[dd])

        if self.default_rule_["determined"]:
            s += "- Else:\n"
            s += "\t* Recourse Rule (Acc. = {}/{} = {:.1%}, Cost = {:.4})\n".format(
                self.default_rule_["acc"].shape[0],
                self.default_rule_["cov"].shape[0],
                self.default_rule_["acc"].shape[0] / self.default_rule_["cov"].shape[0],
                self.default_rule_["cst"],
            )
            for dd in self.rule_to_discretized_feature_[
                self.default_rule_["consequent"]
            ]:
                s += "\t\t* {}\n".format(self.discretized_feature_names_[dd])

        if self.print_objective_:
            if self.feasible_:
                s += "\n"
                s += "### Objective Value\n"
                if self.objective_ == "origin":
                    s += "- Obj. = - {} * {} + {} * {} - {} * {:.4} = {:.4}\n".format(
                        self.lambda_acc,
                        self.inc_,
                        self.lambda_cov,
                        self.cov_,
                        self.lambda_cst,
                        self.cst_,
                        self.obj_,
                    )
                else:
                    s += "- Obj. = {} * {} + {} * {} - {} * {:.4} = {:.4}\n".format(
                        self.lambda_acc,
                        self.acc_,
                        self.lambda_cov,
                        self.cov_,
                        self.lambda_cst,
                        self.cst_,
                        self.obj_,
                    )
            else:
                s += "- No feasible solution.\n"
        return s

    def to_markdown(self):
        s = "| | Rule | Action |\n"
        s += "| :---: | --- | --- |\n"
        for i, r in enumerate(self.R_):
            s += "| Recourse <br> rule {} <br> (probability: {:.1%}) ".format(
                i + 1, r["probability"]
            )
            s += "| If {} |".format(
                self.rule_names_[r["antecedent"]].replace("AND", "<br> AND")
            )
            for dd in self.rule_to_discretized_feature_[r["consequent"]][:-1]:
                s += " {} <br> AND".format(self.discretized_feature_names_[dd])
            s += " {} |\n".format(
                self.discretized_feature_names_[
                    self.rule_to_discretized_feature_[r["consequent"]][-1]
                ]
            )
        if self.default_rule_["determined"]:
            s += "| Default <br> rule | Else |"
            for dd in self.rule_to_discretized_feature_[
                self.default_rule_["consequent"]
            ][:-1]:
                s += " {} <br> AND".format(self.discretized_feature_names_[dd])
            s += " {} |\n".format(
                self.discretized_feature_names_[
                    self.rule_to_discretized_feature_[self.default_rule_["consequent"]][
                        -1
                    ]
                ]
            )
        return s

    def predict(self, X, max_rule=-1):
        ret = []
        X_rule = self.rule_miner_.transform(X)
        if max_rule > 0:
            R = self.R_[: min(self.max_rule_, max_rule)]
        else:
            R = self.R_
        for x, x_rule in zip(X, X_rule):
            a = np.zeros(self.D_)
            R_x = [r for r in R if x_rule[r["antecedent"]] == 1]
            if len(R_x) > 0:
                r_x = R_x[np.argmax([r["probability"] for r in R_x])]
                for d, v in zip(r_x["feature"], r_x["value"]):
                    a[d] = v - x[d]
            elif self.default_rule_["determined"]:
                r_x = self.default_rule_
                for d, (l, u) in zip(r_x["feature"], r_x["value"]):  # noqa: E741
                    if x[d] < l and self.feature_constraints_[d] in ["", "INC"]:
                        v = l
                    elif x[d] >= u and self.feature_constraints_[d] in ["", "DEC"]:
                        v = u - 1 if self.feature_types_[d] == "I" else u - self.tol_
                    else:
                        v = 0
                    a[d] = v - x[d]
            ret += [a]
        return np.array(ret)

    def cost(self, X, cost_type="TLPS", max_rule=-1):
        A = self.predict(X, max_rule=max_rule)
        return np.array(
            [self.cost_.compute(x, a, cost_type=cost_type) for x, a in zip(X, A)]
        ).mean()

    def loss(self, X, target=0, max_rule=-1):
        A = self.predict(X, max_rule=max_rule)
        return (self.mdl_.predict(X + A) != target).mean()

    def uncover(self, X, max_rule=-1):
        X_rule = self.rule_miner_.transform(X)
        if max_rule > 0:
            R = self.R_[: min(self.max_rule_, max_rule)]
        else:
            R = self.R_
        return np.mean(
            [
                len([r for r in R if x_rule[r["antecedent"]] == 1]) == 0
                for x_rule in X_rule
            ]
        )

    def conflict(self, X, max_rule=-1):
        X_rule = self.rule_miner_.transform(X)
        if max_rule > 0:
            R = self.R_[: min(self.max_rule_, max_rule)]
        else:
            R = self.R_
        ret = []
        for x_rule in X_rule:
            R_x = [r for r in R if x_rule[r["antecedent"]] == 1]
            if len(R_x) > 0:
                ret.append(len(R_x))
        # return np.mean(ret)
        return np.mean(np.array(ret) > 1)

    def tradeoff(self, X, cost_type="TLPS", gamma=1.0):
        ret = {"n_rules": [], "cost": [], "loss": [], "obj.": []}
        for n in range(1, self.max_rule_ + 1):
            ret["n_rules"].append(n)
            c = self.cost(X, cost_type=cost_type, max_rule=n)
            l = self.loss(X, max_rule=n)  # noqa: E741
            ret["cost"].append(c)
            ret["loss"].append(l)
            ret["obj."].append(c + gamma * l)
        return ret


def _check(dataset="h", model="L"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from utils import DatasetHelper

    np.random.seed(0)

    print("# Learing Actionable Recourse Summary")
    D = DatasetHelper(dataset=dataset)
    print("* Dataset:", D.dataset_fullname)
    for d in range(D.n_features):
        print(
            "\t* x_{:<2}: {} ({}:{})".format(
                d + 1, D.feature_names[d], D.feature_types[d], D.feature_constraints[d]
            )
        )

    if model == "L":
        print("* Classifier: LogisticRegression")
        from sklearn.linear_model import LogisticRegression

        mdl = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", class_weight="balanced"
        )
        print("\t* C: {}".format(mdl.C))
        print("\t* penalty: {}".format(mdl.penalty))
    elif model == "F":
        print("* Classifier: RandomForest")
        from sklearn.ensemble import RandomForestClassifier

        mdl = RandomForestClassifier(
            n_estimators=100, max_leaf_nodes=16, class_weight="balanced"
        )
        print("\t* n_estimators: {}".format(mdl.n_estimators))
        print("\t* max_leaf_nodes: {}".format(mdl.max_leaf_nodes))
    elif model == "M":
        print("* Classifier: MultiLayerPerceptron")
        from sklearn.neural_network import MLPClassifier

        mdl = MLPClassifier(
            hidden_layer_sizes=(50,), max_iter=500, activation="relu", alpha=0.0001
        )
        print("\t* hidden_layer_size: {}".format(mdl.hidden_layer_sizes[0]))
        print("\t* activation: {}".format(mdl.activation))
    elif model == "X":
        print("* Classifier: LightGBM")
        from lightgbm import LGBMClassifier

        mdl = LGBMClassifier(n_estimators=100, num_leaves=16, class_weight="balanced")
        print("\t* n_estimators: {}".format(mdl.n_estimators))
        print("\t* num_leaves: {}".format(mdl.num_leaves))
    elif model == "T":
        print("* Classifier: TabNet")
        from utils import MyTabNetClassifier

        mdl = MyTabNetClassifier(D.feature_types, verbose=0, class_weight="balanced")

    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr)
    X = X_tr[mdl.predict(X_tr) == 1]

    ares = AReS(
        mdl,
        X_tr,
        max_rule=8,
        max_rule_length=4,
        minimum_support=MINSUP[dataset],
        discretization_bins=10,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
    )

    print("* Candidate Rules:")
    for p in range(0, 3):
        print("\t* r_{}: {}".format(p, ares.rule_names_[p]))
    print(".\n.\n.")
    for p in range(ares.P_ - 3, ares.P_):
        print("\t* r_{}: {}".format(p, ares.rule_names_[p]))
    print()

    print("## Learning AReS")
    ares = ares.fit(
        X,
        max_change_num=3,
        cost_type="MPS",
        lambda_acc=0.01,
        lambda_cov=1.0,
        lambda_cst=0.01,
    )
    print("* Parameters:")
    print("\t* lambda_acc: {}".format(ares.lambda_acc))
    print("\t* lambda_cov: {}".format(ares.lambda_cov))
    print("\t* lambda_cst: {}".format(ares.lambda_cst))
    print("\t* minimum support: {}".format(ares.rule_miner_.minsup_))
    print("\t* discretization bins: {}".format(ares.rule_miner_.fd_.bins))
    print("\t* pre-processing time[s]: {}".format(ares.preprocess_time_))
    print("\t* max rule: {}".format(ares.max_rule_))
    print("\t* max rule length: {}".format(ares.max_rule_length_))
    print("\t* Time[s]:", ares.time_)
    print()

    print("### Learned AReS")
    print(ares)

    print("### Score:")
    print("- Train:")
    print("\t- cost: {}".format(ares.cost(X, cost_type="MPS")))
    print("\t- loss: {}".format(ares.loss(X)))
    print("\t- uncover: {}".format(ares.uncover(X)))
    print("\t- conflict: {}".format(ares.conflict(X)))
    tradeoff = ares.tradeoff(X, cost_type="MPS", gamma=1.0)
    print("\t- trade-off:")
    print("\t\t- cost:", tradeoff["cost"])
    print("\t\t- loss:", tradeoff["loss"])
    print("\t\t- obj.:", tradeoff["obj."])

    X = X_ts[mdl.predict(X_ts) == 1]
    print("- Test:")
    print("\t- cost: {}".format(ares.cost(X, cost_type="MPS")))
    print("\t- loss: {}".format(ares.loss(X)))
    print("\t- uncover: {}".format(ares.uncover(X)))
    print("\t- conflict: {}".format(ares.conflict(X)))
    tradeoff = ares.tradeoff(X, cost_type="MPS", gamma=1.0)
    print("\t- trade-off:")
    print("\t\t- cost:", tradeoff["cost"])
    print("\t\t- loss:", tradeoff["loss"])
    print("\t\t- obj.:", tradeoff["obj."])


def _check_tuning(dataset="h", model="L", gamma=1.0):
    from sklearn.linear_model import LogisticRegression
    from utils import DatasetHelper

    np.random.seed(0)

    print("# Learing Actionable Recourse Summary")
    D = DatasetHelper(dataset=dataset)
    print("* Dataset:", D.dataset_fullname)
    for d in range(D.n_features):
        print(
            "\t* x_{:<2}: {} ({}:{})".format(
                d + 1, D.feature_names[d], D.feature_types[d], D.feature_constraints[d]
            )
        )

    if model == "L":
        print("* Classifier: LogisticRegression")
        from sklearn.linear_model import LogisticRegression

        mdl = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
        print("\t* C: {}".format(mdl.C))
        print("\t* penalty: {}".format(mdl.penalty))
    elif model == "X":
        print("* Classifier: LightGBM")
        from lightgbm import LGBMClassifier

        mdl = LGBMClassifier(n_estimators=100, num_leaves=16, class_weight="balanced")
        print("\t* n_estimators: {}".format(mdl.n_estimators))
        print("\t* num_leaves: {}".format(mdl.num_leaves))
    elif model == "T":
        print("* Classifier: TabNet")
        from utils import MyTabNetClassifier

        mdl = MyTabNetClassifier(D.feature_types, verbose=0, class_weight="balanced")

    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr)
    X = X_tr[mdl.predict(X_tr) == 1]
    X_vl = X_ts[mdl.predict(X_ts) == 1]

    ares = AReS(
        mdl,
        X_tr,
        max_rule=8,
        max_rule_length=4,
        minimum_support=MINSUP[dataset],
        discretization_bins=10,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        feature_categories=D.feature_categories,
        feature_constraints=D.feature_constraints,
        target_name=D.target_name,
        target_labels=D.target_labels,
    )
    print("* Candidate Rules:")
    for p in range(0, 3):
        print("\t* r_{}: {}".format(p, ares.rule_names_[p]))
    print(".\n.\n.")
    for p in range(ares.P_ - 3, ares.P_):
        print("\t* r_{}: {}".format(p, ares.rule_names_[p]))
    print()

    print("## Tuning Hyper-Parameters")
    ares = ares.tuning(
        X,
        max_change_num=3,
        cost_type="MPS",
        gamma=gamma,
        after_fit=True,
        lambda_acc=[0.01, 0.1, 1.0, 10.0, 100.0],
        lambda_cov=[1.0],
        lambda_cst=[0.01, 0.1, 1.0, 10.0, 100.0],
    )

    print("## Learning AReS")
    print("* Parameters:")
    print("\t* lambda_acc: {}".format(ares.lambda_acc))
    print("\t* lambda_cov: {}".format(ares.lambda_cov))
    print("\t* lambda_cst: {}".format(ares.lambda_cst))
    print("\t* minimum support: {}".format(ares.rule_miner_.minsup_))
    print("\t* discretization bins: {}".format(ares.rule_miner_.fd_.bins))
    print("\t* pre-processing time[s]: {}".format(ares.preprocess_time_))
    print("\t* max rule: {}".format(ares.max_rule_))
    print("\t* max rule length: {}".format(ares.max_rule_length_))
    print("\t* Time[s]:", ares.time_)
    print()

    print("### Learned AReS")
    print(ares)

    print("### Score:")
    print("- Train:")
    print("\t- cost: {}".format(ares.cost(X, cost_type="MPS")))
    print("\t- loss: {}".format(ares.loss(X)))
    print("\t- obj.: {}".format(ares.cost(X, cost_type="MPS") + gamma * ares.loss(X)))
    print("\t- uncover: {}".format(ares.uncover(X)))
    print("\t- conflict: {}".format(ares.conflict(X)))
    X = X_ts[mdl.predict(X_ts) == 1]
    print("- Test:")
    print("\t- cost: {}".format(ares.cost(X, cost_type="MPS")))
    print("\t- loss: {}".format(ares.loss(X)))
    print("\t- obj.: {}".format(ares.cost(X, cost_type="MPS") + gamma * ares.loss(X)))
    print("\t- uncover: {}".format(ares.uncover(X)))
    print("\t- conflict: {}".format(ares.conflict(X)))


MINSUP = {"g": 0.05, "i": 0.05, "h": 0.1, "w": 0.1, "d": 0.01}
if __name__ == "__main__":
    # _check(dataset='g')

    _check_tuning(model="X", dataset="i", gamma=1.0)
    _check_tuning(model="X", dataset="g", gamma=1.0)
    _check_tuning(model="T", dataset="i", gamma=1.0)
    _check_tuning(model="T", dataset="g", gamma=1.0)
