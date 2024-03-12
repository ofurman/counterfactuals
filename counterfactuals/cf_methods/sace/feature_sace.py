import numpy as np

from collections import defaultdict
from itertools import combinations

from counterfactuals.sace.sace import SACE


class FeatureSACE(SACE):
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
        tol=0.01,
        nbr_intervals=5,
        nbr_features_to_test=2,
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
            tol,
        )
        self.nbr_intervals = nbr_intervals
        self.nbr_features_to_test = nbr_features_to_test

        self.min_val = None
        self.max_val = None
        self.gap = None

    def fit(self, b, X):
        super().fit(b, X)

        self.min_val = defaultdict(dict)
        self.max_val = defaultdict(dict)
        self.gap = defaultdict(dict)

        classes = np.unique(self.y)
        for y_val in classes:
            X_cfc = self.X[self.y != y_val]

            if len(X_cfc) > 0:
                for fi in self.variable_features:
                    self.min_val[y_val][fi] = np.min(X_cfc[:, fi])
                    self.max_val[y_val][fi] = np.max(X_cfc[:, fi])
                    if len(np.unique(X_cfc[:, fi])) <= 2:
                        self.gap[y_val][fi] = (
                            self.max_val[y_val][fi] - self.min_val[y_val][fi]
                        )
                    else:
                        self.gap[y_val][fi] = (
                            self.max_val[y_val][fi] - self.min_val[y_val][fi]
                        ) / self.nbr_intervals

    def binary_search(self, vx, vcf, cf, y_cf, j, eps=0.01):
        # print(vx, vcf, cf, y_cf, j)
        v_min = min(vx, vcf)
        v_max = max(vx, vcf)
        last_y_ok_val = vcf
        iter_count = 0
        max_iter = 20
        while (v_max - v_min) > eps and iter_count < max_iter:
            v_mid = v_min + np.abs((v_max - v_min) / 2)
            v_mid = self._round_value(v_mid)
            cfc = cf.copy()
            cfc[j] = v_mid
            y_cfc = self._predict(cfc.reshape(1, -1))
            if y_cfc == y_cf:
                vx, vcf = vx, v_mid
                last_y_ok_val = vcf
            else:
                vx, vcf = v_mid, vcf

            v_min = min(vx, vcf)
            v_max = max(vx, vcf)
            iter_count += 1

            # print(v_max, v_min)

        # print(vx, vcf, cf, y_cf, j, last_y_ok_val, 'AAAAAA')
        return last_y_ok_val

    def refine_values(self, x, cf_list, eps=0.01):
        diff = np.abs(cf_list - x)
        y_cf_list = self._predict(cf_list)
        for i in range(len(cf_list)):
            cf = cf_list[i]
            y_cf = y_cf_list[i]
            for j in self.continuous_features:
                if diff[i][j] == 0:
                    continue

                nfv = self.binary_search(x[:, j][0], cf[j], cf, y_cf, j, eps)
                cf[j] = nfv

        return cf_list

    def get_counterfactuals(
        self,
        x,
        k=5,
        y_desiderd=None,
        constrain_into_ranges=True,
        search_diversity=False,
        refine_values=True,
    ):
        # x = x.reshape(1, -1)
        x = np.expand_dims(x, 0)
        y_val = self.b.predict(x)[0]

        cf_score = dict()
        cf_list_set = set()
        for nbr_features_tested in range(1, self.nbr_features_to_test + 1):
            for vf in combinations(self.variable_features, nbr_features_tested):
                # cfc = x.copy()
                cfc = x.copy() if not self.pooler else self.pooler.transform(x)
                self.__update_feature_values(
                    x,
                    y_val,
                    cfc,
                    vf,
                    0,
                    len(vf),
                    cf_score,
                    y_desiderd,
                    constrain_into_ranges,
                    cf_list_set,
                )

        if len(cf_score) > k and search_diversity:
            cf_list = self._get_diverse(cf_score, k)
        else:
            cf_list = self._get_closest(cf_score, k)

        if refine_values and len(cf_list) and not self.pooler:
            cf_list = self.refine_values(x, cf_list)

        if self.pooler:
            if len(cf_list) > 0:
                cf_list = self.pooler.inverse_transform(cf_list)
            else:
                cf_list = np.array([])

        return cf_list

    def __update_feature_values(
        self,
        x,
        y_val,
        cfc,
        vf,
        fi,
        max_nf,
        cf_score,
        y_desiderd,
        constrain_into_ranges,
        cf_list_set,
    ):
        if fi < max_nf:
            min_val = self.min_val[y_val][vf[fi]]
            max_val = self.max_val[y_val][vf[fi]]
            gap = self.gap[y_val][vf[fi]]
            if max_val + gap > min_val:
                for new_val in np.arange(min_val, max_val + gap, gap):
                    # cfc[:, vf[fi]] = new_val
                    cfc[:, vf[fi]] = self._round_value(new_val)
                    self.__update_feature_values(
                        x,
                        y_val,
                        cfc,
                        vf,
                        fi + 1,
                        max_nf,
                        cf_score,
                        y_desiderd,
                        constrain_into_ranges,
                        cf_list_set,
                    )
        else:
            cfc_tuple = tuple(cfc.flatten())
            if cfc_tuple in cf_list_set:
                return

            cf_list_set.add(cfc_tuple)

            y_cfc = self._predict(cfc)[0]
            if y_desiderd is None and y_cfc != y_val or y_cfc == y_desiderd:
                if not self._respect_ranges(cfc):
                    if constrain_into_ranges:
                        cfc = self._contrain_into_ranges(cfc)
                    else:
                        return

                if not self._respect_categorical_features(cfc):
                    return

                # nx = self.scaler.transform(x)
                nx = (
                    self.scaler.transform(x)
                    if not self.pooler
                    else self.pooler.transform(x)
                )
                ncfc = self.scaler.transform(cfc)
                score = self.cdist(ncfc, nx, metric=self.metric, w=self.weights)
                cf_score[len(cf_score)] = (score, cfc.copy().flatten())

    def get_prototypes(
        self,
        x,
        k=5,
        beta=0.5,
        constrain_into_ranges=True,
        search_diversity=False,
        refine_values=True,
    ):
        # x = x.reshape(1, -1)
        x = np.expand_dims(x, 0)
        y_val = self.b.predict(x)[0]
        y_prob = self.b.predict_proba(x)[:, y_val][0]

        pr_score = dict()
        pr_list_set = set()
        for nbr_features_tested in range(1, self.nbr_features_to_test + 1):
            for vf in combinations(self.variable_features, nbr_features_tested):
                # prc = x.copy()
                prc = x.copy() if not self.pooler else self.pooler.transform(x)
                self.__update_feature_values_pr(
                    x,
                    y_val,
                    y_prob,
                    prc,
                    vf,
                    0,
                    len(vf),
                    pr_score,
                    beta,
                    constrain_into_ranges,
                    pr_list_set,
                )

        if len(pr_score) > k and search_diversity:
            pr_list = self._get_diverse(pr_score, k)
        else:
            pr_list = self._get_closest(pr_score, k)

        if refine_values and len(pr_list) and not self.pooler:
            pr_list = self.refine_values(x, pr_list)

        if self.pooler:
            pr_list = self.pooler.inverse_transform(pr_list)

        return pr_list

    def __update_feature_values_pr(
        self,
        x,
        y_val,
        y_prob,
        prc,
        vf,
        fi,
        max_nf,
        pr_score,
        beta,
        constrain_into_ranges,
        pr_list_set,
    ):
        if fi < max_nf:
            min_val = self.min_val[y_val][vf[fi]]
            max_val = self.max_val[y_val][vf[fi]]
            gap = self.gap[y_val][vf[fi]]
            if max_val + gap > min_val:
                for new_val in np.arange(min_val, max_val + gap, gap):
                    # prc[:, vf[fi]] = new_val
                    prc[:, vf[fi]] = self._round_value(new_val)
                    self.__update_feature_values_pr(
                        x,
                        y_val,
                        y_prob,
                        prc,
                        vf,
                        fi + 1,
                        max_nf,
                        pr_score,
                        beta,
                        constrain_into_ranges,
                        pr_list_set,
                    )
        else:
            prc_tuple = tuple(prc.flatten())
            if prc_tuple in pr_list_set:
                return

            pr_list_set.add(prc_tuple)
            y_prc = self._predict(prc)[0]
            y_prc_prob = self._predict_proba(prc)[:, y_prc][0]

            if y_prc == y_val and y_prc_prob > y_prob:
                if not self._respect_ranges(prc):
                    if constrain_into_ranges:
                        prc = self._contrain_into_ranges(prc)
                    else:
                        return

                if not self._respect_categorical_features(prc):
                    return

                nx = self.scaler.transform(x)
                n_prc = self.scaler.transform(prc)
                score = self._calculate_prototype_score(nx, n_prc, beta=beta)
                pr_score[len(pr_score)] = (score, prc.copy().flatten())
