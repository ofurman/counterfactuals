import numpy as np

from functools import partial

from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier

from scipy.spatial.distance import hamming

from counterfactuals.sace.sace import SACE


def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d**2) / kernel_width**2))


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (
        inner_tree.children_left[index] == TREE_LEAF
        and inner_tree.children_right[index] == TREE_LEAF
    )


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (
        is_leaf(inner_tree, inner_tree.children_left[index])
        and is_leaf(inner_tree, inner_tree.children_right[index])
        and (decisions[index] == decisions[inner_tree.children_left[index]])
        and (decisions[index] == decisions[inner_tree.children_right[index]])
    ):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        # print("Pruned {}".format(index))


def prune_duplicate_leaves(dt):
    # Remove leaves if both
    decisions = (
        dt.tree_.value.argmax(axis=2).flatten().tolist()
    )  # Decision for each node
    prune_index(dt.tree_, decisions)


class TreeSACE(SACE):
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
        use_instance_weights=False,
        kernel_width=None,
        min_samples_leaf=0.01,
        max_depth=None,
        closest_in_leaf=False,
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

        self.use_instance_weights = use_instance_weights
        self.kernel_width = kernel_width
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.closest_in_leaf = closest_in_leaf

        self.kernel = None

    def fit(self, b, X):
        super().fit(b, X)

        self.kernel_width = float(
            np.sqrt(self.nbr_variable_features) * 0.75
            if self.kernel_width is None
            else self.kernel_width
        )
        self.kernel = partial(default_kernel, kernel_width=self.kernel_width)

    def get_counterfactuals(
        self,
        x,
        k=5,
        y_desiderd=None,
        constrain_into_ranges=True,
        search_diversity=False,
        alpha=1.0,
        nbr_trees=1,
    ):
        # x = x.reshape(1, -1)
        x = np.expand_dims(x, 0)
        nx = (
            self.scaler.transform(x)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(x))
        )

        y_val = self.b.predict(x)[0]

        if self.use_instance_weights:
            sample_weight = self.__calculate_weights__(nx, self.nX, self.metric)
        else:
            sample_weight = None

        cf_score = dict()
        cf_list_set = set()
        for ti in range(nbr_trees):
            cf_list_scored_ti = self.__get_counterfactuals(
                x,
                nx,
                y_val,
                k,
                y_desiderd,
                constrain_into_ranges,
                alpha,
                sample_weight,
                ti,
            )
            for cfs in cf_list_scored_ti.values():
                cfc_tuple = tuple(cfs[1])
                if cfc_tuple not in cf_list_set:
                    cf_list_set.add(cfc_tuple)
                    cf_score[len(cf_score)] = cfs

        if len(cf_score) > k and search_diversity:
            cf_list = self._get_diverse(cf_score, k)
        else:
            cf_list = self._get_closest(cf_score, k)

        if self.pooler:
            if len(cf_list) > 0:
                cf_list = self.pooler.inverse_transform(cf_list)
            else:
                cf_list = np.array([])

        return cf_list

    def __get_counterfactuals(
        self,
        x,
        nx,
        y_val,
        k=5,
        y_desiderd=None,
        constrain_into_ranges=True,
        alpha=1.0,
        sample_weight=None,
        seed=None,
    ):
        self.dt = DecisionTreeClassifier(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=None,
            random_state=seed,
        )
        self.dt.fit(self.X, self.y, sample_weight=sample_weight)
        prune_duplicate_leaves(self.dt)
        self.feature = self.dt.tree_.feature
        self.threshold = self.dt.tree_.threshold
        self.tree_depth = self.dt.get_depth() + 1

        # y_val_dt = self.dt.predict(x)[0]
        # if y_val != y_val_dt:
        #     print('Disagreement between black box and decision tree: %s, %s' % (y_val, y_val_dt))
        # raise Exception('Disagreement between black box and decision tree: %s, %s' % (y_val, y_val_dt))

        if y_desiderd is None:
            cond = self.y != y_val
        else:
            cond = self.y == y_desiderd

        X_cfc = self.X[cond]
        cf_score = dict()
        if len(X_cfc) == 0:
            return cf_score
        # print(self.dt.predict(X_cfc), 'AAAA')
        # print('--->', len(X_cfc), y_val, np.unique(self.y))
        # node_indicator = self.dt.decision_path(X_cfc)
        leave_id = self.dt.apply(X_cfc)

        X_cfcm, node_indicator_m, leave_id_m = self.__get_mean_leaf_cf_infos(
            X_cfc, leave_id
        )
        if not self.pooler:
            node_index_path_x, leave_id_x = self.__get_x_infos(x)
            feature_lower_upper_x = self.__get_feature_boundaries(x, node_index_path_x)
        else:
            node_index_path_x, leave_id_x = self.__get_x_infos(self.pooler.transform(x))
            feature_lower_upper_x = self.__get_feature_boundaries(
                self.pooler.transform(x), node_index_path_x
            )

        for cf_idx, l_idx in zip(range(len(X_cfcm)), np.unique(leave_id_m)):
            node_index_path_cf = node_indicator_m.indices[
                node_indicator_m.indptr[cf_idx] : node_indicator_m.indptr[cf_idx + 1]
            ]

            cfcm = X_cfcm[cf_idx].reshape(1, -1)
            feature_lower_upper_cf = self.__get_feature_boundaries(
                cfcm, node_index_path_x
            )
            nvf_changed = False
            for node_id in node_index_path_x:
                f = self.feature[node_id]
                if f in self.non_variable_features:
                    # se viene utilizzata variabile azionabile controllare che vengono rispettate le condizioni di x
                    cond_ok = (
                        feature_lower_upper_cf[f][0] >= feature_lower_upper_x[f][0]
                        and feature_lower_upper_cf[f][1] <= feature_lower_upper_x[f][1]
                    )
                    # print(f, self.feature_names[f], cfcm[0, self.feature[node_id]], feature_lower_upper_cf[f], feature_lower_upper_x[f], cond_ok)

                    if not cond_ok:
                        nvf_changed = True

            if not nvf_changed:
                if len(node_index_path_cf) < self.tree_depth:
                    pad = self.tree_depth - len(node_index_path_cf)
                    node_index_path_cf = np.concatenate(
                        [node_index_path_cf, np.array([-1] * pad)]
                    )

                dist = hamming(node_index_path_x, node_index_path_cf)

                X_cfc_l = X_cfc[leave_id == l_idx]
                nX_cfc_l = self.scaler.transform(X_cfc_l)
                score = alpha * dist + (1 - alpha) * 1 / (len(X_cfc_l) / len(X_cfc))

                # cfc_o = x.copy()
                cfc_o = x.copy() if not self.pooler else self.pooler.transform(x)
                if self.closest_in_leaf:
                    cfl_idx = np.argmin(
                        self.cdist(nx, nX_cfc_l, metric=self.metric, w=self.weights)
                    )
                    cfc = X_cfc_l[cfl_idx].copy().reshape(1, -1)
                    cfc[:, self.non_variable_features] = cfc_o[
                        :, self.non_variable_features
                    ]
                else:
                    cfc = cfcm.copy()
                    cfc[:, self.non_variable_features] = cfc_o[
                        :, self.non_variable_features
                    ]

                y_cfc = self._predict(cfc)[0]

                if y_desiderd is None and y_cfc != y_val or y_cfc == y_desiderd:
                    if not self._respect_ranges(cfc):
                        if constrain_into_ranges:
                            cfc = self._contrain_into_ranges(cfc)
                        else:
                            continue

                    if not self._respect_categorical_features(cfc):
                        continue

                    cf_score[l_idx] = (score, cfc.flatten())

        return cf_score

    def get_prototypes(
        self,
        x,
        k=5,
        beta=0.5,
        constrain_into_ranges=True,
        search_diversity=False,
        alpha=1.0,
        nbr_trees=1,
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

        if self.use_instance_weights:
            sample_weight = self.__calculate_weights__(nx, self.nX, self.metric)
        else:
            sample_weight = None

        pr_list_set = set()
        pr_score = dict()
        for ti in range(nbr_trees):
            pr_list_scored_ti = self.__get_prototypes(
                x,
                nx,
                y_val,
                y_prob,
                k,
                constrain_into_ranges,
                alpha,
                sample_weight,
                ti,
                beta,
            )
            for prs in pr_list_scored_ti.values():
                prc_tuple = tuple(prs[1])
                if prc_tuple not in pr_list_set:
                    pr_list_set.add(prc_tuple)
                    pr_score[len(pr_score)] = prs

        if len(pr_score) > k and search_diversity:
            pr_list = self._get_diverse(pr_score, k)
        else:
            pr_list = self._get_closest(pr_score, k)

        if self.pooler:
            pr_list = self.pooler.inverse_transform(pr_list)

        return pr_list

    def __get_prototypes(
        self,
        x,
        nx,
        y_val,
        y_prob,
        k=5,
        constrain_into_ranges=False,
        alpha=1.0,
        sample_weight=None,
        seed=None,
        beta=0.5,
    ):
        self.dt = DecisionTreeClassifier(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=None,
            random_state=seed,
        )
        self.dt.fit(self.X, self.y, sample_weight=sample_weight)
        prune_duplicate_leaves(self.dt)
        self.feature = self.dt.tree_.feature
        self.threshold = self.dt.tree_.threshold
        self.tree_depth = self.dt.get_depth() + 1

        # y_val_dt = self.dt.predict(x)[0]
        # if y_val != y_val_dt:
        #     raise Exception('Disagreement between black box and decision tree: %s, %s' % (y_val, y_val_dt))

        cond = self.y == y_val

        X_prc = self.X[cond]
        leave_id = self.dt.apply(X_prc)

        X_prcm, node_indicator_m, leave_id_m = self.__get_mean_leaf_cf_infos(
            X_prc, leave_id
        )
        if not self.pooler:
            node_index_path_x, leave_id_x = self.__get_x_infos(x)
            feature_lower_upper_x = self.__get_feature_boundaries(x, node_index_path_x)
        else:
            node_index_path_x, leave_id_x = self.__get_x_infos(self.pooler.transform(x))
            feature_lower_upper_x = self.__get_feature_boundaries(
                self.pooler.transform(x), node_index_path_x
            )

        pr_score = dict()
        for pr_idx, l_idx in zip(range(len(X_prcm)), np.unique(leave_id_m)):
            node_index_path_pr = node_indicator_m.indices[
                node_indicator_m.indptr[pr_idx] : node_indicator_m.indptr[pr_idx + 1]
            ]

            prcm = X_prcm[pr_idx].reshape(1, -1)
            feature_lower_upper_pr = self.__get_feature_boundaries(
                prcm, node_index_path_x
            )
            nvf_changed = False
            for node_id in node_index_path_x:
                f = self.feature[node_id]
                if f in self.non_variable_features:
                    cond_ok = (
                        feature_lower_upper_pr[f][0] >= feature_lower_upper_x[f][0]
                        and feature_lower_upper_pr[f][1] <= feature_lower_upper_x[f][1]
                    )

                    if not cond_ok:
                        nvf_changed = True

            if not nvf_changed:
                if len(node_index_path_pr) < self.tree_depth:
                    pad = self.tree_depth - len(node_index_path_pr)
                    node_index_path_pr = np.concatenate(
                        [node_index_path_pr, np.array([-1] * pad)]
                    )

                dist = hamming(node_index_path_x, node_index_path_pr)

                X_prc_l = X_prc[leave_id == l_idx]
                nX_prc_l = self.scaler.transform(X_prc_l)
                score = alpha * dist + (1 - alpha) * 1 / (len(X_prc_l) / len(X_prc))

                if self.closest_in_leaf:
                    cfl_idx = np.argmin(
                        self.cdist(nx, nX_prc_l, metric=self.metric, w=self.weights)
                    )
                    prc = X_prc_l[cfl_idx].copy().reshape(1, -1)
                else:
                    prc = prcm.copy()

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

                    pr_score[l_idx] = (score, prc.flatten())

        return pr_score

    def __get_feature_boundaries(self, x, node_index_path):
        feature_lower_upper = dict()
        for node_id in node_index_path:
            f = self.feature[node_id]
            if f == -2:
                continue
            threshold_sign = "<=" if x[0, f] <= self.threshold[node_id] else ">"

            if f not in feature_lower_upper:
                feature_lower_upper[f] = [-np.inf, np.inf]

            if threshold_sign == "<=":
                if self.threshold[node_id] < feature_lower_upper[f][1]:
                    feature_lower_upper[f][1] = self.threshold[node_id]
            else:
                if self.threshold[node_id] > feature_lower_upper[f][0]:
                    feature_lower_upper[f][0] = self.threshold[node_id]

        return feature_lower_upper

    def __get_mean_leaf_cf_infos(self, X_cfc, leave_id):
        X_cfcm = list()
        for lid in np.unique(leave_id):
            X_cfcm.append(np.mean(X_cfc[leave_id == lid], axis=0))
        X_cfcm = np.array(X_cfcm)
        node_indicator_m = self.dt.decision_path(X_cfcm)
        leave_id_m = self.dt.apply(X_cfcm)
        return X_cfcm, node_indicator_m, leave_id_m

    def __get_x_infos(self, x):
        node_indicator_x = self.dt.decision_path(x)
        node_index_path_x = node_indicator_x.indices[
            node_indicator_x.indptr[0] : node_indicator_x.indptr[0 + 1]
        ]
        # feature_path_x = self.feature[node_index_path_x][:-1]
        if len(node_index_path_x) < self.tree_depth:
            pad = self.tree_depth - len(node_index_path_x)
            node_index_path_x = np.concatenate(
                [node_index_path_x, np.array([-2] * pad)]
            )

        leave_id_x = self.dt.apply(x)

        return node_index_path_x, leave_id_x

    def __calculate_weights__(self, x, X, metric):
        # distances = cdist(x, X, metric=metric, w=self.weights).ravel()
        distances = self.cdist(x, X, metric=metric, w=self.weights).ravel()
        weights = self.kernel(distances)
        return weights
