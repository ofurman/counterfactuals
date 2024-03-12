import numpy as np

from counterfactuals.sace.sace import SACE


class NeighborSACE(SACE):
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

    def fit(self, b, X):
        super().fit(b, X)

        if self.random_samples and self.random_samples < len(self.X):
            index_random_samples = np.random.choice(
                range(len(self.X)), self.random_samples
            )
            self.X = self.X[index_random_samples]
            self.y = self.y[index_random_samples]

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
        dists = self.cdist(nx, nX_cfc, metric=self.metric, w=self.weights)

        cf_score = dict()
        cf_list_set = set()
        for cf_idx in np.argsort(dists)[0]:
            cfc = x.copy() if not self.pooler else self.pooler.transform(x)
            cfc[:, self.variable_features] = X_cfc[cf_idx, self.variable_features]
            y_cfc = self._predict(cfc)[0]
            # print(y_cfc, y_val, y_cfc != y_val, dists[0][cf_idx])
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
                    cf_list_set.add(cfc_tuple)
                    score = dists[0][cf_idx]
                    cf_score[len(cf_score)] = (score, cfc.flatten())
                    if len(cf_score) == k:
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
        # x = x.reshape(1, -1)
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
