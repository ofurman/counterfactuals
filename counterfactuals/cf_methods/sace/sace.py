import numpy as np

from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist, pdist, squareform

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from counterfactuals.cf_methods.sace.dummy_scaler import DummyScaler


class SACE(ABC):
    def __init__(
        self,
        variable_features=None,
        weights=None,
        metric="euclidean",
        feature_names=None,
        continuous_features=None,
        categorical_features_index_lists=None,
        normalize=True,
        pooler=None,
        tol=0.01,
    ):
        self.variable_features = variable_features
        self.weights = weights
        self.metric = metric
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features_index_lists = categorical_features_index_lists
        self.normalize = normalize
        self.pooler = pooler
        self.tol = tol

        self.b = None
        self.X = None
        self.y = None
        self.nbr_features = None
        self.non_variable_features = None
        self.nbr_variable_features = None
        self.scaler = None
        self.nX = None

    @abstractmethod
    def fit(self, b, X):
        """

        :param b: black box predict function
        :param X: training set of b
        :param V: list of features to vary
        """

        self.b = b
        self.y = self.b.predict(X)

        if self.pooler:
            X_p = self.pooler.transform(X)
            self.X = X_p
        else:
            self.X = X

        self.nbr_features = self.X.shape[1]
        self.variable_features = (
            self.variable_features
            if self.variable_features is not None
            else np.arange(self.nbr_features).tolist()
        )
        self.non_variable_features = [
            i for i in range(self.nbr_features) if i not in self.variable_features
        ]
        self.nbr_variable_features = len(self.variable_features)

        self.scaler = StandardScaler() if self.normalize else DummyScaler()
        self.scaler.fit(self.X)
        self.nX = self.scaler.transform(self.X)

        self.__init_cont_cat_features(self.continuous_features)
        self.__detect_ranges()

    @abstractmethod
    def get_counterfactuals(
        self,
        x,
        k=5,
        y_desiderd=None,
        constrain_into_ranges=True,
        search_diversity=False,
    ):
        pass

    @abstractmethod
    def get_prototypes(
        self, x, k=5, beta=0.5, constrain_into_ranges=True, search_diversity=False
    ):
        pass

    def _predict(self, X):
        if self.pooler:
            X_up = self.pooler.inverse_transform(X)
            return self.b.predict(X_up)
        return self.b.predict(X)

    def _predict_proba(self, X):
        if self.pooler:
            X_up = self.pooler.inverse_transform(X)
            return self.b.predict_proba(X_up)
        return self.b.predict_proba(X)

    def __init_cont_cat_features(self, continuous_features):
        if isinstance(self.metric, str):
            self.continuous_features = (
                continuous_features  # np.arange(self.nbr_variable_features).tolist()
            )
            self.categorical_features = []
            self.cdist = self._cdist0
        else:
            self.continuous_features = continuous_features
            self.categorical_features = [
                i
                for i in np.arange(self.nbr_variable_features)
                if i not in continuous_features
            ]
            self.cdist = self._cdist
        self.nbr_cont_features = len(self.continuous_features)
        if self.categorical_features_index_lists:
            self.nbr_cate_features_real = len(self.categorical_features_index_lists)
        else:
            self.nbr_cate_features_real = 0
        self.nbr_variable_features_real = (
            self.nbr_cont_features + self.nbr_cate_features_real
        )

    def __detect_ranges(self):
        self.ranges = dict()
        for i in self.variable_features:
            self.ranges[i] = [np.min(self.X[:, i]), np.max(self.X[:, i])]

    def _respect_ranges(self, x):
        for i in self.variable_features:
            if x[:, i] < self.ranges[i][0] or x[:, i] > self.ranges[i][1]:
                return False
        return True

    def _respect_categorical_features(self, x):
        if self.categorical_features_index_lists is None:
            return True

        for idx_list in self.categorical_features_index_lists:
            if np.sum(x[:, idx_list]) != 1.0:
                return False
            # if self.input_normalized:
            #     if np.abs(np.sum(x[:, idx_list])) < 1.0:
            #         print(x[:, idx_list], np.sum(x[:, idx_list]))
            #         return False
            # else:
            #     if np.sum(x[:, idx_list]) != 0.0:
            #         return False

        return True

    def _contrain_into_ranges(self, x):
        for i in self.variable_features:
            if x[:, i] < self.ranges[i][0]:
                x[:, i] = self.ranges[i][0]
            if x[:, i] > self.ranges[i][1]:
                x[:, i] = self.ranges[i][1]
        return x

    def _cdist(self, XA, XB, metric=("euclidean", "jaccard"), w=None):
        metric_continuous = metric[0]
        metric_categorical = metric[1]
        dist_continuous = cdist(
            XA[:, self.continuous_features],
            XB[:, self.continuous_features],
            metric=metric_continuous,
            w=w,
        )
        dist_categorical = cdist(
            XA[:, self.categorical_features],
            XB[:, self.categorical_features],
            metric=metric_categorical,
            w=w,
        )
        ratio_continuous = self.nbr_cont_features / self.nbr_variable_features_real
        ratio_categorical = (
            self.nbr_cate_features_real / self.nbr_variable_features_real
        )
        dist = ratio_continuous * dist_continuous + ratio_categorical * dist_categorical

        return dist

    def _cdist0(self, XA, XB, metric="euclidean", w=None):
        dist_continuous = cdist(
            XA[:, self.continuous_features],
            XB[:, self.continuous_features],
            metric=metric,
            w=w,
        )
        dist = dist_continuous

        return dist

    def _calculate_prototype_score(self, x, pr, beta=0.5):
        # d = self.cdist(x[:, self.variable_features], pr[:, self.variable_features],
        #                metric=self.metric, w=self.weights)[0]
        d = self.cdist(x, pr, metric=self.metric, w=self.weights)[0]
        if d == 0.0:
            return np.inf
        l = cdist(self._predict_proba(x), self._predict_proba(pr), metric="cosine")[0]  # noqa: E741
        if l == 0.0:
            return np.inf
        score = beta * d + (1.0 - beta) * 1.0 / l
        return score

    def _get_closest(self, cf_score, k):
        cf_list = list()
        for cf_idx, _ in sorted(
            cf_score.items(), key=lambda cf: cf[1][0], reverse=False
        )[:k]:
            cf_list.append(cf_score[cf_idx][1])
        cf_list = np.array(cf_list)
        return cf_list

    def _get_diverse(self, cf_score, k):
        kmeans = KMeans(n_clusters=k)
        X = np.array([cf[1] for cf in cf_score.values()])
        kmeans.fit(X)
        labels = np.unique(kmeans.labels_)
        cf_list = list()
        for label in labels:
            cfl = X[kmeans.labels_ == label]
            if len(cfl) <= 2:
                idx = 0
            else:
                idx = np.argmin(np.sum(squareform(pdist(cfl)), axis=0))
            cf_list.append(cfl[idx])
        cf_list = np.array(cf_list)
        return cf_list

    def _round_value(self, val):
        decimals = int(-np.log10(self.tol))
        return np.round(val, decimals)

    # def _filter_similar(self, x, cf_list):
    #     decimals = int(-np.log10(self.tol))
    #     cf_list_ = set()
    #     for cf in cf_list:
    #         cft = cf.copy()
    #         for i in self.variable_features:
    #             if x[i] != cft[i]:
    #                 cft[i] = np.round(cft[i], decimals)
    #         cf_list_.add(tuple(cft))
    #     cf_list_ = np.array([np.array(cf) for cf in cf_list_])
    #     return cf_list_
