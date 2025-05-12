import numpy as np
from pyfpgrowth import find_frequent_patterns


class FeatureDiscretizer:
    def __init__(self, bins=5, onehot=True):
        self.bins = bins
        self.onehot = onehot

    def fit(self, X, feature_names=[], feature_types=[]):
        self.X_ = X
        origin_feature_names = (
            feature_names
            if len(feature_names) == X.shape[1]
            else ["x_{}".format(d) for d in range(X.shape[1])]
        )
        origin_feature_types = (
            feature_types if len(feature_types) == X.shape[1] else ["C"] * X.shape[1]
        )

        self.thresholds = []
        self.feature_names = []
        self.feature_types = []
        for d in range(X.shape[1]):
            if origin_feature_types[d] == "B":
                self.thresholds.append([])
            else:
                threshold_d = []
                for threshold in np.asarray(
                    np.percentile(
                        X[:, d],
                        np.linspace(
                            0, 100, self.bins if self.onehot else self.bins + 1
                        ),
                    )
                ):
                    if origin_feature_types[d] == "I":
                        threshold = int(threshold)
                    else:
                        threshold = np.around(threshold, 4)
                    if threshold in threshold_d:
                        continue
                    if np.unique(X[:, d] < threshold).size == 1:
                        continue
                    threshold_d.append(threshold)
                self.thresholds.append(threshold_d)

        for d in range(X.shape[1]):
            if origin_feature_types[d] == "B":
                self.feature_names += [origin_feature_names[d]]
                self.feature_types += ["B"]
            else:
                if self.onehot:
                    self.feature_names += [
                        "{}<{}".format(origin_feature_names[d], self.thresholds[d][0])
                    ]
                    self.feature_types += ["R"]
                    for i in range(len(self.thresholds[d]) - 1):
                        self.feature_names += [
                            "{}<={}<{}".format(
                                self.thresholds[d][i],
                                origin_feature_names[d],
                                self.thresholds[d][i + 1],
                            )
                        ]
                        self.feature_types += ["R"]
                    self.feature_names += [
                        "{}<={}".format(self.thresholds[d][-1], origin_feature_names[d])
                    ]
                    self.feature_types += ["R"]
                else:
                    for i in range(len(self.thresholds[d])):
                        self.feature_names += [
                            "{}<{}".format(
                                origin_feature_names[d], self.thresholds[d][i]
                            )
                        ]
                        self.feature_types += ["R"]

        self.D_ = len(self.feature_names)
        self.origin_feature_names = origin_feature_names
        self.origin_feature_types = origin_feature_types
        return self

    def transform(self, X):
        X_bin = np.zeros([X.shape[0], self.D_])
        i = 0
        for d in range(X.shape[1]):
            if self.origin_feature_types[d] == "B":
                X_bin[:, i] = X[:, d]
                # print(i, d)
                i += 1
            else:
                if self.onehot:
                    X_bin[:, i] = (X[:, d] < self.thresholds[d][0]).astype(int)
                    i += 1
                    for j in range(len(self.thresholds[d]) - 1):
                        X_bin[
                            np.where(
                                (X[:, d] >= self.thresholds[d][j])
                                & (X[:, d] < self.thresholds[d][j + 1])
                            )[0],
                            i,
                        ] = 1
                        i += 1
                    X_bin[:, i] = (X[:, d] >= self.thresholds[d][-1]).astype(int)
                    i += 1
                else:
                    for j in range(len(self.thresholds[d])):
                        X_bin[:, i] = (X[:, d] < self.thresholds[d][j]).astype(int)
                        i += 1
        return X_bin


class FrequentRuleMiner:
    def __init__(self, minsup=0.05, discretization=False):
        self.minsup_ = minsup
        self.discretization_ = discretization

    def __str__(self):
        s = ""
        for l, rule in enumerate(self.rule_names_):
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

    def miningFrequentRules(self, X):
        N = X.shape[0]
        threshold = self.minsup_ if self.minsup_ in range(1, N) else N * self.minsup_
        transaction = self.OnehotsToTransactions(X)
        patterns = find_frequent_patterns(transaction, threshold)
        return list(patterns.keys())

    def fit(
        self,
        X,
        feature_names=[],
        feature_types=[],
        discretization_bins=5,
        discretization_onehot=True,
        save_file="",
    ):
        if self.discretization_:
            self.fd_ = FeatureDiscretizer(
                bins=discretization_bins, onehot=discretization_onehot
            ).fit(X, feature_names=feature_names, feature_types=feature_types)
            self.feature_names = self.fd_.feature_names
            self.feature_types = self.fd_.feature_types
            X = self.fd_.transform(X)
        else:
            self.feature_names = feature_names
            self.feature_types = feature_types

        self.D_ = X.shape[1]
        self.rules_ = self.miningFrequentRules(X)
        self.length_ = len(self.rules_)

        # rule indicator matrix (D * length)
        self.Z_ = np.zeros([self.length_, self.D_], dtype=int)
        for l, rule in enumerate(self.rules_):
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


if __name__ == "__main__":
    from utils import DatasetHelper

    np.random.seed(0)

    D = DatasetHelper(dataset="g", feature_prefix_index=False)
    X_tr, X_ts, y_tr, y_ts = D.train_test_split()

    frm = FrequentRuleMiner(minsup=0.5)
    frm = frm.fit(
        X_tr,
        feature_names=D.feature_names,
        feature_types=D.feature_types,
        discretization=True,
        discretization_bins=5,
    )
    X_ts_rule = frm.transform(X_ts[y_ts == 1], discretization=True)
    for d, feat in enumerate(frm.feature_names):
        print(d, feat)
    for l, rule in enumerate(frm.rule_names_):
        print(l, rule)
    print(X_ts_rule[0], X_ts_rule.mean())
