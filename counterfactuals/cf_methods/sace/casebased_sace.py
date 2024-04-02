import numpy as np

from scipy.spatial.distance import cdist
from collections import defaultdict

from counterfactuals.cf_methods.sace.sace import SACE


class CaseBasedSACE(SACE):
    def __init__(
        self,
        variable_features,
        weights=None,
        metric="euclidean",
        feature_names=None,
        continuous_features=None,
        categorical_features_lists=None,
        normalize=True,
        pooler=None,
        random_samples=None,
        diff_features=2,
        tolerance=0.01,
    ):
        super().__init__(
            variable_features,
            weights,
            metric,
            feature_names,
            continuous_features,
            categorical_features_lists,
            normalize,
            pooler,
        )

        self.random_samples = random_samples
        self.diff_features = diff_features
        self.tolerance = tolerance

        self.diff_dist = None
        self.XC_x = None
        self.XC_list = None

    def _diff_dist(self, XA, XB):
        dist_continuous = (
            (
                cdist(
                    XA[:, self.continuous_features],
                    XB[:, self.continuous_features],
                    metric="euclidean",
                )
                > self.tolerance
            )
            .flatten()
            .astype(np.float64)
        )

        dist_categorical = (
            np.sum(
                np.abs(
                    XA[:, self.categorical_features] - XB[:, self.categorical_features]
                ),
                axis=1,
            )
            / 2
        )

        dist = dist_continuous + dist_categorical

        return dist

    def _diff_dist0(self, XA, XB):
        dist_continuous = (
            (
                cdist(
                    XA[:, self.continuous_features],
                    XB[:, self.continuous_features],
                    metric="euclidean",
                )
                > self.tolerance
            )
            .flatten()
            .astype(np.float)
        )
        dist = dist_continuous

        return dist

    def fit(self, b, X):
        super().fit(b, X)

        if self.random_samples and self.random_samples < len(self.X):
            index_random_samples = np.random.choice(
                range(len(self.X)), self.random_samples
            )
            self.X = self.X[index_random_samples]
            self.y = self.y[index_random_samples]

        if isinstance(self.metric, str):
            self.diff_dist = self._diff_dist0
        else:
            self.diff_dist = self._diff_dist

        self.XC_x = defaultdict(list)
        self.XC_list = defaultdict(list)
        nX = (
            self.scaler.transform(self.X)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(self.X))
        )

        for nx, label in zip(nX, self.y):
            nX_cfc = nX[self.y != label]
            idx_list = np.argwhere(self.y != label).flatten()
            dists = self.diff_dist(np.expand_dims(nx, 0), nX_cfc)
            xc_list = list()
            for i, d in zip(idx_list, dists):
                if d <= self.diff_features:
                    xc_list.append(i)
            self.XC_x[label].append(nx)
            self.XC_list[label].append(xc_list)

        for label in self.XC_x:
            self.XC_x[label] = np.array(self.XC_x[label])

    def get_counterfactuals(
        self,
        x,
        k=5,
        y_desiderd=None,
        constrain_into_ranges=True,
        search_diversity=False,
    ):
        # x = x.reshape(1, -1)
        x = np.expand_dims(x, 0)
        nx = (
            self.scaler.transform(x)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(x))
        )
        y_val = self.b.predict(x)[0]

        XC_sc = self.XC_x[y_val]
        XC_cf_list = self.XC_list[y_val]
        dists = self.cdist(nx, XC_sc, metric=self.metric, w=self.weights)

        cf_score = dict()
        cf_list_set = set()
        flag = False
        for sc_idx in np.argsort(dists)[0]:
            cf_idx_list = XC_cf_list[sc_idx]
            flag1 = False
            for cf_idx in cf_idx_list:
                cfc = x.copy() if not self.pooler else self.pooler.transform(x)
                cfc[:, self.variable_features] = self.X[cf_idx, self.variable_features]
                y_cfc = self._predict(cfc)[0]
                # print(sc_idx, y_val, y_cfc)
                if y_desiderd is None and y_cfc != y_val or y_cfc == y_desiderd:
                    if not self._respect_ranges(cfc):
                        if constrain_into_ranges:
                            cfc = self._contrain_into_ranges(cfc)
                        else:
                            continue

                    if not self._respect_categorical_features(cfc):
                        continue

                    cfc_tuple = tuple(cfc.flatten())
                    if cfc_tuple not in cf_list_set:
                        flag1 = True
                        cf_list_set.add(cfc_tuple)
                        score = dists[0][sc_idx]
                        cf_score[len(cf_score)] = (score, cfc.flatten())
                        if len(cf_score) == k:
                            flag = True
                            break

                else:  # point 4 of algorithm
                    ncfc = (
                        self.scaler.transform(cfc)
                        if not self.pooler
                        else self.scaler.transform(self.pooler.transform(cfc))
                    )

                    if y_desiderd is None:
                        cond = self.y != y_val
                    else:
                        cond = self.y == y_desiderd

                    X_cfc = self.X[cond]
                    nX_cfc = (
                        self.scaler.transform(X_cfc)
                        if not self.pooler
                        else self.scaler.transform(self.pooler.transform(X_cfc))
                    )

                    dists_cfc = self.cdist(
                        ncfc, nX_cfc, metric=self.metric, w=self.weights
                    )
                    for cf_idx2 in np.argsort(dists_cfc)[0]:
                        cfc2 = x.copy() if not self.pooler else self.pooler.transform(x)
                        cfc2[:, self.variable_features] = X_cfc[
                            cf_idx2, self.variable_features
                        ]
                        y_cfc2 = self._predict(cfc2)[0]
                        # print(sc_idx, y_val, y_cfc2)
                        if (
                            y_desiderd is None
                            and y_cfc2 != y_val
                            or y_cfc2 == y_desiderd
                        ):
                            if not self._respect_ranges(cfc2):
                                if constrain_into_ranges:
                                    cfc2 = self._contrain_into_ranges(cfc2)
                                else:
                                    continue

                            if not self._respect_categorical_features(cfc2):
                                continue

                            cfc_tuple = tuple(cfc2.flatten())
                            if cfc_tuple not in cf_list_set:
                                flag1 = True
                                cf_list_set.add(cfc_tuple)
                                score = dists[0][sc_idx]
                                cf_score[len(cf_score)] = (score, cfc2.flatten())
                                if len(cf_score) == k:
                                    flag = True
                                    break

                        if flag1:
                            break

                if flag or flag1:
                    break

            if flag:
                break

        if len(cf_score) > k and search_diversity:
            cf_list = self._get_diverse(cf_score, k)
        else:
            cf_list = self._get_closest(cf_score, k)

        if self.pooler:
            cf_list = self.pooler.inverse_transform(cf_list)

        return cf_list

    def get_prototypes(
        self, x, k=5, beta=0.5, constrain_into_ranges=True, search_diversity=False
    ):
        x = np.expand_dims(x, 0)
        nx = (
            self.scaler.transform(x)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(x))
        )
        y_val = self.b.predict(x)[0]
        y_prob = self.b.predict_proba(x)[:, y_val][0]

        cond = self.y == y_val

        X_prc = self.X[cond]
        nX_prc = (
            self.scaler.transform(X_prc)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(X_prc))
        )

        dists = self.cdist(nx, nX_prc, metric=self.metric, w=self.weights)

        pr_score = dict()
        pr_list_set = set()
        for cf_idx in np.argsort(dists)[0]:
            prc = x.copy() if not self.pooler else self.pooler.transform(x)
            prc[:, self.variable_features] = X_prc[cf_idx, self.variable_features]
            y_prc = self._predict(prc)[0]
            y_prc_prob = self._predict_proba(prc)[:, y_prc][0]

            if y_prc == y_val and y_prc_prob > y_prob:
                if not self._respect_ranges(prc):
                    if constrain_into_ranges:
                        prc = self._contrain_into_ranges(prc)
                    else:
                        continue

                if not self._respect_categorical_features(prc):
                    continue

                prc_tuple = tuple(prc.flatten())
                if prc_tuple not in pr_list_set:
                    pr_list_set.add(prc_tuple)
                    n_prc = self.scaler.transform(prc)
                    score = self._calculate_prototype_score(nx, n_prc, beta=beta)
                    pr_score[len(pr_score)] = (score, prc.flatten())

        if len(pr_score) > k and search_diversity:
            pr_list = self._get_diverse(pr_score, k)
        else:
            pr_list = self._get_closest(pr_score, k)

        if self.pooler:
            pr_list = self.pooler.inverse_transform(pr_list)

        return pr_list
