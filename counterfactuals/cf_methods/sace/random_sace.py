import numpy as np

from counterfactuals.sace.sace import SACE


class RandomSACE(SACE):
    def __init__(
        self,
        variable_features=None,
        weights=None,
        metric="euclidean",
        feature_names=None,
        continuous_features=None,
        categorical_features_lists=None,
        normalize=True,
        pooler=None,
        n_attempts=100,
        n_max_attempts=10000,
        proba=0.5,
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
        self.n_attempts = n_attempts
        self.n_max_attempts = n_max_attempts
        self.proba = proba

        self.f_values = None

    def fit(self, b, X):
        super().fit(b, X)
        self.f_values = dict()
        for f_idx in self.variable_features:
            self.f_values[f_idx] = np.unique(self.X[:, f_idx])

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

        cf_list_set = set()
        cf_score = dict()
        for i in range(self.n_max_attempts):
            cfc = x.copy() if not self.pooler else self.pooler.transform(x)
            for f_idx in self.variable_features:
                if np.random.random() >= self.proba:
                    cfc[:, f_idx] = np.random.choice(self.f_values[f_idx])
                    y_cfc = self._predict(cfc)[0]
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

                            nc = self.scaler.transform(cfc)
                            score = self.cdist(
                                nc.reshape(1, -1),
                                nx,
                                metric=self.metric,
                                w=self.weights,
                            ).flatten()[0]
                            cf_score[len(cf_score)] = (score, cfc.flatten())

                            if len(cf_score) == self.n_attempts:
                                break

            if len(cf_score) == self.n_attempts:
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

        pr_score = dict()
        pr_list_set = set()
        for i in range(self.n_max_attempts):
            prc = x.copy() if not self.pooler else self.pooler.transform(x)
            for f_idx in self.variable_features:
                if np.random.random() >= self.proba:
                    prc[:, f_idx] = np.random.choice(self.f_values[f_idx])
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
                            npre = self.scaler.transform(prc)
                            score = self._calculate_prototype_score(
                                npre.reshape(1, -1), nx, beta=beta
                            )
                            pr_score[len(pr_score)] = (score, prc.flatten())

                            if len(pr_score) == self.n_attempts:
                                break

            if len(pr_score) == self.n_attempts:
                break

        if len(pr_score) > k and search_diversity:
            pr_list = self._get_diverse(pr_score, k)
        else:
            pr_list = self._get_closest(pr_score, k)

        if self.pooler:
            pr_list = self.pooler.inverse_transform(pr_list)

        return pr_list
