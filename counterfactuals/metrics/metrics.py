import numpy as np

from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_abs_deviation

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KernelDensity 

class DummyScaler:

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X
    
def valid_cf(y, y_cf):
    return y_cf != y

def number_valid_cf(y, y_cf):
    val = np.sum(valid_cf(y, y_cf))
    return val

def perc_valid_cf(y, y_cf):
    n_val = number_valid_cf(y, y_cf=y_cf)
    res = n_val / len(y)
    return res

def actionable_cf(X, X_cf, actionable_features: list):
    # TODO: rewrite
    assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
    actionable = np.all((X == X_cf)[:, actionable_features], axis=1)
    return actionable

def number_actionable_cf(X, X_cf, actionable_features: list):
    assert X.shape == X_cf.shape
    number_actionable = np.sum(actionable_cf(X, X_cf, actionable_features), axis=1)
    return number_actionable


def perc_actionable_cf(X, X_cf, actionable_features: list):
    assert X.shape == X_cf.shape
    n_val = number_actionable_cf(X, X_cf, actionable_features)
    res = n_val / len(X_cf)
    return res


def valid_actionable_cf(X, X_cf, y, y_cf, actionable_features):
    valid = valid_cf(y, y_cf)
    actionable = actionable_cf(X, X_cf, actionable_features)

    assert valid.shape == actionable.shape
    return np.logical_and(valid, actionable)

def number_valid_actionable_cf(X, X_cf, y, y_cf, actionable_features):
    return np.sum(valid_actionable_cf(X, X_cf, y, y_cf, actionable_features))


def perc_valid_actionable_cf(X, X_cf, y, y_cf, actionable_features):
    n_val = number_valid_actionable_cf(X, X_cf, y, y_cf, actionable_features)
    return n_val / len(y)


def number_violations_per_cf(X, X_cf, actionable_features: list):
    assert X.shape == X_cf.shape
    res = np.sum((X == X_cf)[:, actionable_features], axis=1)
    return res


def avg_number_violations_per_cf(X, X_cf, actionable_features):
    return np.mean(number_violations_per_cf(X, X_cf, actionable_features))


def avg_number_violations(X, X_cf, actionable_features):
    val = np.sum(number_violations_per_cf(X, X_cf, actionable_features))
    number_cf, number_features = X_cf.shape
    return val / (number_cf * number_features)

def sparsity(X, X_cf, actionable_features=None):
    number_cf, number_features = X_cf.shape
    val = X != X_cf
    if actionable_features is not None:
        val = val[:, actionable_features]
    val = np.sum(val)
    return val / (number_cf * number_features)


def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def continuous_distance(X, X_cf, continuous_features, metric='euclidean', X_all=None, agg='mean', _diag=True):
    assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
    if metric == 'mad':
        mad = median_abs_deviation(X_all[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        metric = _mad_cityblock

    dist = cdist(X[:, continuous_features], X_cf[:, continuous_features], metric=metric)

    dist = np.diag(dist) if _diag else dist
    agg_funcs = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'no': lambda x: x
    }
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    return agg_funcs[agg](dist)


def categorical_distance(X, X_cf, categorical_features, metric='jaccard', agg=None, _diag=True):
    assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
    dist = cdist(X[:, categorical_features], X_cf[:, categorical_features], metric=metric)
    dist = np.diag(dist) if _diag else dist
    agg_funcs = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'no': lambda x: x
    }
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    return agg_funcs[agg](dist)


def distance_l2_jaccard(X, X_cf, continuous_features, categorical_features, ratio_cont=None):
    number_features = X_cf.shape[1]
    dist_cont = continuous_distance(X, X_cf, continuous_features, metric='euclidean', X_all=None, agg='mean')
    dist_cate = categorical_distance(X, X_cf, categorical_features, metric='jaccard', agg='mean')
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / number_features
        ratio_categorical = len(categorical_features) / number_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_continuous
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def distance_mad_hamming(X, X_cf, continuous_features, categorical_features, X_all, ratio_cont=None, agg=None, diag=True):
    number_features = X_cf.shape[1]
    dist_cont = continuous_distance(X, X_cf, continuous_features, metric='mad', X_all=X_all, agg=agg, _diag=diag)
    dist_cate = categorical_distance(X, X_cf, categorical_features, metric='hamming', agg=agg, _diag=diag)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / number_features
        ratio_categorical = len(categorical_features) / number_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def number_changes_per_cf(X, X_cf, continuous_features, agg='mean'):
    assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
    result = np.sum(X[:, continuous_features] == X_cf[:, continuous_features], axis=1)
    agg_funcs = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'sum': np.sum,
        'no': lambda x: x,
    }
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    return agg_funcs[agg](result)

def avg_number_changes(X, X_cf, continuous_features):
    number_cf, number_features = X_cf.shape[1]
    val = number_changes_per_cf(X, X_cf, continuous_features, agg='sum')
    return val / (number_cf * number_features)

def plausibility(X_test, X_cf, y_test, continuous_features_all, categorical_features_all, X_train, ratio_cont=None):
    dist_neighb = distance_mad_hamming(X_test, X_test, continuous_features_all, categorical_features_all, X_train, ratio_cont=ratio_cont, agg='no', diag=False)
    dist_neighb[y_test == 0, y_test == 0] = np.inf
    dist_neighb[y_test == 1, y_test == 1] = np.inf
    idx_neighb = np.argmin(dist_neighb, axis=0)
    dist_neighb = distance_mad_hamming(X_test[idx_neighb], X_cf, continuous_features_all, categorical_features_all, X_train, ratio_cont=ratio_cont, agg='mean', diag=True)
    return dist_neighb

def delta_proba(x, cf_list, classifier, agg=None):
    y_val = classifier.predict_proba(x)
    y_cf = classifier.predict_proba(cf_list)
    deltas = np.abs(y_cf - y_val)

    if agg is None or agg == 'mean':
        return np.mean(deltas)

    if agg == 'max':
        return np.max(deltas)

    if agg == 'min':
        return np.min(deltas)
    
def kde_density(X_train, y_train, Xs, Xs_cfs, ys, return_x_log_density=False):
    log_density = []
    log_density_cfs = []

    for y in np.unique(y_train):
        X_train_y = X_train[y_train == y]
        Xs_y = Xs[ys == y]
        Xs_cfs_y = Xs_cfs[ys != y]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_train_y)
        
        log_dens_cfs = kde.score_samples(Xs_cfs_y)
        log_density_cfs.append(log_dens_cfs)
        if return_x_log_density:
            log_dens = kde.score_samples(Xs_y)
            log_density.append(log_dens)
        
    if return_x_log_density:
        return np.mean(np.hstack(log_density)), np.mean(np.hstack(log_density_cfs))
    else:
        return np.mean(np.hstack(log_density_cfs))
    
def evaluate_cf(disc_model, X, X_cf, continuous_features, categorical_features, X_train, y_train, X_test, y_test):
    
    ys_disc_pred = disc_model.predict(X)
    ys_cfs_disc_pred = disc_model.predict(X_cf)
    X = X[ys_cfs_disc_pred != ys_disc_pred]
    X_cf = X_cf[ys_cfs_disc_pred != ys_disc_pred]

    disc_model.predict(X)

    metrics = {
        # "valid_cf_gen_orig": perc_valid_cf(ys_orig, y_cf=ys_cfs_gen_pred),
        # "valid_cf_disc_orig": perc_valid_cf(ys_orig, y_cf=ys_cfs_disc_pred),
        # "valid_cf_gen": perc_valid_cf(ys_orig_gen_pred, y_cf=ys_cfs_gen_pred),
        "valid_cf_disc": perc_valid_cf(ys_disc_pred, y_cf=ys_cfs_disc_pred),
        # "perc_valid_actionable_cf": perc_valid_actionable_cf(X=dataset.X_test[:100], X_cf=Xs_cfs, y=ys_orig_pred, y_cf=ys_cfs_pred,
        #                                                      actionable_features=dataset.actionable_features),
        "dissimilarity_proximity_categorical_hamming": categorical_distance(
            X=X, X_cf=X_cf, categorical_features=categorical_features, metric="hamming", agg="mean"
        ),
        "dissimilarity_proximity_categorical_jaccard": categorical_distance(
            X=X, X_cf=X_cf, categorical_features=categorical_features, metric="jaccard", agg="mean"
        ),
        "dissimilarity_proximity_continuous_manhatan": continuous_distance(
            X=X, X_cf=X_cf, continuous_features=continuous_features, metric="cityblock", X_all=X_test
        ),
        "dissimilarity_proximity_continuous_euclidean": continuous_distance(
            X=X, X_cf=X_cf, continuous_features=continuous_features, metric="euclidean", X_all=X_test
        ),
        "dissimilarity_proximity_continuous_mad": continuous_distance(
            X=X, X_cf=X_cf, continuous_features=continuous_features, metric="mad", X_all=X_test
        ),
        "distance_l2_jaccard": distance_l2_jaccard(
            X=X,
            X_cf=X_cf,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
        ),
        "distance_mad_hamming": distance_mad_hamming(
            X=X,
            X_cf=X_cf,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            X_all=X_test,
            agg="mean",
        ),
        "plausibility": plausibility(
            X,
            X_cf,
            disc_model.predict(X),
            continuous_features_all=continuous_features,
            categorical_features_all=categorical_features,
            X_train=X_train,
            ratio_cont=None,
        ),
        # "flow_log_density": np.mean(np.concatenate([log_p_zeros[ys_orig == 0], log_p_ones[ys_orig == 1]])),
        "kde_log_density": kde_density(X_train=X_train, y_train=y_train, Xs=X, Xs_cfs=X_cf, ys=disc_model.predict(X)),
        "sparsity": sparsity(X, X_cf),
    }
    return metrics

# def continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=None):
#     if metric == 'mad':
#         mad = median_abs_deviation(X[:, continuous_features], axis=0)
#         mad = np.array([v if v != 0 else 1.0 for v in mad])
#         def _mad_cityblock(u, v):
#             return mad_cityblock(u, v, mad)
#         metric = _mad_cityblock
#     dist = pdist(cf_list[:, continuous_features], metric=metric)

#     agg_funcs = {
#         'mean': np.mean,
#         'max': np.max,
#         'min': np.min,
#         'sum': np.sum,
#         'no': lambda x: x,
#     }
#     assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
#     return agg_funcs[agg](dist)


# def categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=None):
#     dist = pdist(cf_list[:, categorical_features], metric=metric)
#     agg_funcs = {
#         'mean': np.mean,
#         'max': np.max,
#         'min': np.min,
#         'sum': np.sum,
#         'no': lambda x: x,
#     }
#     assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
#     return agg_funcs[agg](dist)


# def diversity_l2j(cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
#     number_features = cf_list.shape[1]
#     dist_cont = continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
#     dist_cate = categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=agg)
#     if ratio_cont is None:
#         ratio_continuous = len(continuous_features) / number_features
#         ratio_categorical = len(categorical_features) / number_features
#     else:
#         ratio_continuous = ratio_cont
#         ratio_categorical = 1.0 - ratio_cont
#     dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
#     return dist


# def diversity_mh(cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
#     number_features = cf_list.shape[1]
#     dist_cont = continuous_diversity(cf_list, continuous_features, metric='mad', X=X, agg=agg)
#     dist_cate = categorical_diversity(cf_list, categorical_features, metric='hamming', agg=agg)
#     if ratio_cont is None:
#         ratio_continuous = len(continuous_features) / number_features
#         ratio_categorical = len(categorical_features) / number_features
#     else:
#         ratio_continuous = ratio_cont
#         ratio_categorical = 1.0 - ratio_cont
#     dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
#     return dist


# def count_diversity(cf_list, features, number_features, continuous_features):
#     number_cf = cf_list.shape[0]
#     number_changes = 0
#     for i in range(number_cf):
#         for j in range(i+1, number_cf):
#             for k in features:
#                 if cf_list[i][k] != cf_list[j][k]:
#                     number_changes += 1 if j in continuous_features else 0.5
#     return number_changes / (number_cf * number_cf * number_features)


# # piu alto e' meglio conta variet' tra cf
# def count_diversity_all(cf_list, number_features, continuous_features):
#     return count_diversity(cf_list, range(cf_list.shape[1]), number_features, continuous_features)


# def euclidean_jaccard(x, A, continuous_features, categorical_features, ratio_cont=None):
#     number_features = A.shape[1]
#     dist_cont = cdist(x.reshape(1, -1)[:, continuous_features], A[:, continuous_features], metric='euclidean')
#     dist_cate = cdist(x.reshape(1, -1)[:, categorical_features], A[:, categorical_features], metric='jaccard')
#     if ratio_cont is None:
#         ratio_continuous = len(continuous_features) / number_features
#         ratio_categorical = len(categorical_features) / number_features
#     else:
#         ratio_continuous = ratio_cont
#         ratio_categorical = 1.0 - ratio_cont
#     dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
#     return dist


# def accuracy_knn_sklearn(x, cf_list, b, X_test, continuous_features, categorical_features, scaler, test_size=5):
#     clf = KNeighborsClassifier(n_neighbors=1)
#     X_train = np.vstack([x.reshape(1, -1), cf_list])
#     y_train = b.predict(X_train)
#     clf.fit(X_train, y_train)

#     X_test_knn = select_test_knn(x, b, X_test, continuous_features, categorical_features, scaler, test_size)
#     y_test = b.predict(X_test_knn)
#     y_pred = clf.predict(X_test_knn)

#     return accuracy_score(y_test, y_pred)


# def accuracy_knn_dist(x, cf_list, b, X_test, continuous_features, categorical_features, scaler, test_size=5):

#     X_train = np.vstack([x.reshape(1, -1), cf_list])
#     y_train = b.predict(X_train)

#     nX_train = scaler.transform(X_train)

#     X_test_knn, nX_test_knn = select_test_knn(x, b, X_test, continuous_features, categorical_features, scaler,
#                                               test_size, get_normalized=True)
#     y_test = b.predict(X_test_knn)

#     y_pred = list()
#     for nx_test in nX_test_knn:
#         dist = euclidean_jaccard(nx_test, nX_train, continuous_features, categorical_features)
#         idx = np.argmin(dist)
#         y_pred.append(y_train[idx])

#     return accuracy_score(y_test, y_pred)


# def select_test_knn(x, b, X_test, continuous_features, categorical_features, scaler, test_size=5, get_normalized=False):
#     y_val = b.predict(x.reshape(1, -1))
#     y_test = b.predict(X_test)
#     nx = scaler.transform(x.reshape(1, -1))
#     nX_test = scaler.transform(X_test)
#     dist_f = euclidean_jaccard(nx, nX_test[y_test == y_val], continuous_features, categorical_features)
#     dist_cf = euclidean_jaccard(nx, nX_test[y_test != y_val], continuous_features, categorical_features)
#     index_f = np.argsort(dist_f)[0][:test_size].tolist()
#     index_cf = np.argsort(dist_cf)[0][:test_size].tolist()
#     index = np.array(index_f + index_cf)
#     if get_normalized:
#         return X_test[index], nX_test[index]
#     return X_test[index]


# def lof(x, cf_list, X, scaler):
#     X_train = np.vstack([x.reshape(1, -1), X])

#     nX_train = scaler.transform(X_train)
#     ncf_list = scaler.transform(cf_list)

#     clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
#     clf.fit(nX_train)

#     lof_values = clf.predict(ncf_list)
#     return np.mean(np.abs(lof_values))


# # differenza in predict proba con x -> piu alto e' megliio
# def delta_proba(x, cf_list, b, agg=None):
#     y_val = b.predict(x.reshape(1, -1))
#     y_cf = b.predict(cf_list)
#     deltas = np.abs(y_cf - y_val)

#     if agg is None or agg == 'mean':
#         return np.mean(deltas)

#     if agg == 'max':
#         return np.max(deltas)

#     if agg == 'min':
#         return np.min(deltas)


# def delta_proba_img_ts(x, cf_list, b, agg=None):
#     y_val = b.predict(x.reshape((1,) + x.shape))
#     y_cf = np.array([b.predict(cf.reshape((1,) + cf.shape)) for cf in cf_list])
#     deltas = np.abs(y_cf - y_val)

#     if agg is None or agg == 'mean':
#         return np.mean(deltas)

#     if agg == 'max':
#         return np.max(deltas)

#     if agg == 'min':
#         return np.min(deltas)


# Gęstość rozkładu



# def evaluate_cf_list(cf_list, x, classifier, y_val, max_number_cf, variable_features, continuous_features_all,
#                      categorical_features_all, X_train, X_test, ratio_cont, number_features):

#     number_cf_ = len(cf_list)

#     if number_cf_ > 0:
#         scaler = DummyScaler()
#         scaler.fit(X_train)

#         y_pred = classifier.predict(X_test)

#         number_valid_cf_ = number_valid_cf(cf_list, classifier, y_val)
#         perc_valid_cf_ = perc_valid_cf(cf_list, classifier, y_val, k=number_cf_)
#         perc_valid_cf_all_ = perc_valid_cf(cf_list, classifier, y_val, k=max_number_cf)
#         number_actionable_cf_ = number_actionable_cf(x, cf_list, variable_features)
#         perc_actionable_cf_ = perc_actionable_cf(x, cf_list, variable_features, k=number_cf_)
#         perc_actionable_cf_all_ = perc_actionable_cf(x, cf_list, variable_features, k=max_number_cf)
#         number_valid_actionable_cf_ = number_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features)
#         perc_valid_actionable_cf_ = perc_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features, k=number_cf_)
#         perc_valid_actionable_cf_all_ = perc_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features, k=max_number_cf)
#         avg_number_violations_per_cf_ = avg_number_violations_per_cf(x, cf_list, variable_features)
#         avg_number_violations_ = avg_number_violations(x, cf_list, variable_features)

#         plausibility_sum = plausibility(x, classifier, cf_list, X_test, y_pred, continuous_features_all,
#                                         categorical_features_all, X_train, ratio_cont)
#         plausibility_max_number_cf_ = plausibility_sum / max_number_cf
#         plausibility_number_cf_ = plausibility_sum / number_cf_
#         plausibility_number_valid_cf_ = plausibility_sum / number_valid_cf_ if number_valid_cf_ > 0 else plausibility_sum
#         plausibility_number_actionable_cf_ = plausibility_sum / number_actionable_cf_ if number_actionable_cf_ > 0 else plausibility_sum
#         plausibility_number_valid_actionable_cf_ = plausibility_sum / number_valid_actionable_cf_ if number_valid_actionable_cf_ > 0 else plausibility_sum

#         distance_l2_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None)
#         distance_mad_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train)
#         distance_j_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard')
#         distance_h_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming')
#         distance_l2j_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont)
#         distance_mh_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont)

#         distance_l2_min_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None, agg='min')
#         distance_mad_min_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train, agg='min')
#         distance_j_min_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard', agg='min')
#         distance_h_min_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming', agg='min')
#         distance_l2j_min_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='min')
#         distance_mh_min_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='min')

#         distance_l2_max_ = continuous_distance(x, cf_list, continuous_features_all, metric='euclidean', X=None, agg='max')
#         distance_mad_max_ = continuous_distance(x, cf_list, continuous_features_all, metric='mad', X=X_train, agg='max')
#         distance_j_max_ = categorical_distance(x, cf_list, categorical_features_all, metric='jaccard', agg='max')
#         distance_h_max_ = categorical_distance(x, cf_list, categorical_features_all, metric='hamming', agg='max')
#         distance_l2j_max_ = distance_l2j(x, cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='max')
#         distance_mh_max_ = distance_mh(x, cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='max')

#         avg_number_changes_per_cf_ = avg_number_changes_per_cf(x, cf_list, continuous_features_all)
#         avg_number_changes_ = avg_number_changes(x, cf_list, number_features, continuous_features_all)

#         delta_ = delta_proba(x, cf_list, classifier, agg='mean')
#         delta_min_ = delta_proba(x, cf_list, classifier, agg='min')
#         delta_max_ = delta_proba(x, cf_list, classifier, agg='max')

#         if len(cf_list) > 1:
#             diversity_l2_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None)
#             diversity_mad_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train)
#             diversity_j_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard')
#             diversity_h_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming')
#             diversity_l2j_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont)
#             diversity_mh_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont)

#             diversity_l2_min_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None, agg='min')
#             diversity_mad_min_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train, agg='min')
#             diversity_j_min_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard', agg='min')
#             diversity_h_min_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming', agg='min')
#             diversity_l2j_min_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='min')
#             diversity_mh_min_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='min')

#             diversity_l2_max_ = continuous_diversity(cf_list, continuous_features_all, metric='euclidean', X=None, agg='max')
#             diversity_mad_max_ = continuous_diversity(cf_list, continuous_features_all, metric='mad', X=X_train, agg='max')
#             diversity_j_max_ = categorical_diversity(cf_list, categorical_features_all, metric='jaccard', agg='max')
#             diversity_h_max_ = categorical_diversity(cf_list, categorical_features_all, metric='hamming', agg='max')
#             diversity_l2j_max_ = diversity_l2j(cf_list, continuous_features_all, categorical_features_all, ratio_cont, agg='max')
#             diversity_mh_max_ = diversity_mh(cf_list, continuous_features_all, categorical_features_all, X_train, ratio_cont, agg='max')

#         else:
#             diversity_l2_ = 0.0
#             diversity_mad_ = 0.0
#             diversity_j_ = 0.0
#             diversity_h_ = 0.0
#             diversity_l2j_ = 0.0
#             diversity_mh_ = 0.0

#             diversity_l2_min_ = 0.0
#             diversity_mad_min_ = 0.0
#             diversity_j_min_ = 0.0
#             diversity_h_min_ = 0.0
#             diversity_l2j_min_ = 0.0
#             diversity_mh_min_ = 0.0

#             diversity_l2_max_ = 0.0
#             diversity_mad_max_ = 0.0
#             diversity_j_max_ = 0.0
#             diversity_h_max_ = 0.0
#             diversity_l2j_max_ = 0.0
#             diversity_mh_max_ = 0.0

#         count_diversity_cont_ = count_diversity(cf_list, continuous_features_all, number_features, continuous_features_all)
#         count_diversity_cate_ = count_diversity(cf_list, categorical_features_all, number_features, continuous_features_all)
#         count_diversity_all_ = count_diversity_all(cf_list, number_features, continuous_features_all)

#         accuracy_knn_sklearn_ = accuracy_knn_sklearn(x, cf_list, classifier, X_test, continuous_features_all,
#                                                      categorical_features_all, scaler, test_size=5)
#         accuracy_knn_dist_ = accuracy_knn_dist(x, cf_list, classifier, X_test, continuous_features_all,
#                                                categorical_features_all, scaler, test_size=5)

#         lof_ = lof(x, cf_list, X_train, scaler)

#         res = {
#             'number_cf': number_cf_,
#             'number_valid_cf': number_valid_cf_,
#             'perc_valid_cf': perc_valid_cf_,
#             'perc_valid_cf_all': perc_valid_cf_all_,
#             'number_actionable_cf': number_actionable_cf_,
#             'perc_actionable_cf': perc_actionable_cf_,
#             'perc_actionable_cf_all': perc_actionable_cf_all_,
#             'number_valid_actionable_cf': number_valid_actionable_cf_,
#             'perc_valid_actionable_cf': perc_valid_actionable_cf_,
#             'perc_valid_actionable_cf_all': perc_valid_actionable_cf_all_,
#             'avg_number_violations_per_cf': avg_number_violations_per_cf_,
#             'avg_number_violations': avg_number_violations_,
#             'distance_l2': distance_l2_,
#             'distance_mad': distance_mad_,
#             'distance_j': distance_j_,
#             'distance_h': distance_h_,
#             'distance_l2j': distance_l2j_,
#             'distance_mh': distance_mh_,
#             'avg_number_changes_per_cf': avg_number_changes_per_cf_,
#             'avg_number_changes': avg_number_changes_,

#             'distance_l2_min': distance_l2_min_,
#             'distance_mad_min': distance_mad_min_,
#             'distance_j_min': distance_j_min_,
#             'distance_h_min': distance_h_min_,
#             'distance_l2j_min': distance_l2j_min_,
#             'distance_mh_min': distance_mh_min_,

#             'distance_l2_max': distance_l2_max_,
#             'distance_mad_max': distance_mad_max_,
#             'distance_j_max': distance_j_max_,
#             'distance_h_max': distance_h_max_,
#             'distance_l2j_max': distance_l2j_max_,
#             'distance_mh_max': distance_mh_max_,

#             'diversity_l2': diversity_l2_,
#             'diversity_mad': diversity_mad_,
#             'diversity_j': diversity_j_,
#             'diversity_h': diversity_h_,
#             'diversity_l2j': diversity_l2j_,
#             'diversity_mh': diversity_mh_,

#             'diversity_l2_min': diversity_l2_min_,
#             'diversity_mad_min': diversity_mad_min_,
#             'diversity_j_min': diversity_j_min_,
#             'diversity_h_min': diversity_h_min_,
#             'diversity_l2j_min': diversity_l2j_min_,
#             'diversity_mh_min': diversity_mh_min_,

#             'diversity_l2_max': diversity_l2_max_,
#             'diversity_mad_max': diversity_mad_max_,
#             'diversity_j_max': diversity_j_max_,
#             'diversity_h_max': diversity_h_max_,
#             'diversity_l2j_max': diversity_l2j_max_,
#             'diversity_mh_max': diversity_mh_max_,

#             'count_diversity_cont': count_diversity_cont_,
#             'count_diversity_cate': count_diversity_cate_,
#             'count_diversity_all': count_diversity_all_,
#             'accuracy_knn_sklearn': accuracy_knn_sklearn_,
#             'accuracy_knn_dist': accuracy_knn_dist_,
#             'lof': lof_,

#             'delta': delta_,
#             'delta_min': delta_min_,
#             'delta_max': delta_max_,

#             'plausibility_sum': plausibility_sum,
#             'plausibility_max_number_cf': plausibility_max_number_cf_,
#             'plausibility_number_cf': plausibility_number_cf_,
#             'plausibility_number_valid_cf': plausibility_number_valid_cf_,
#             'plausibility_number_actionable_cf': plausibility_number_actionable_cf_,
#             'plausibility_number_valid_actionable_cf': plausibility_number_valid_actionable_cf_,
#         }

#     else:
#         res = {
#             'number_cf': number_cf_,
#             'number_valid_cf': 0.0,
#             'perc_valid_cf': 0.0,
#             'perc_valid_cf_all': 0.0,
#             'number_actionable_cf': 0.0,
#             'perc_actionable_cf': 0.0,
#             'perc_actionable_cf_all': 0.0,
#             'number_valid_actionable_cf': 0.0,
#             'perc_valid_actionable_cf': 0.0,
#             'perc_valid_actionable_cf_all': 0.0,
#             'avg_number_violations_per_cf': np.nan,
#             'avg_number_violations': np.nan,
#             'distance_l2': np.nan,
#             'distance_mad': np.nan,
#             'distance_j': np.nan,
#             'distance_h': np.nan,
#             'distance_l2j': np.nan,
#             'distance_mh': np.nan,
#             'distance_l2_min': np.nan,
#             'distance_mad_min': np.nan,
#             'distance_j_min': np.nan,
#             'distance_h_min': np.nan,
#             'distance_l2j_min': np.nan,
#             'distance_mh_min': np.nan,
#             'distance_l2_max': np.nan,
#             'distance_mad_max': np.nan,
#             'distance_j_max': np.nan,
#             'distance_h_max': np.nan,
#             'distance_l2j_max': np.nan,
#             'distance_mh_max': np.nan,
#             'avg_number_changes_per_cf': np.nan,
#             'avg_number_changes': np.nan,
#             'diversity_l2': np.nan,
#             'diversity_mad': np.nan,
#             'diversity_j': np.nan,
#             'diversity_h': np.nan,
#             'diversity_l2j': np.nan,
#             'diversity_mh': np.nan,
#             'diversity_l2_min': np.nan,
#             'diversity_mad_min': np.nan,
#             'diversity_j_min': np.nan,
#             'diversity_h_min': np.nan,
#             'diversity_l2j_min': np.nan,
#             'diversity_mh_min': np.nan,
#             'diversity_l2_max': np.nan,
#             'diversity_mad_max': np.nan,
#             'diversity_j_max': np.nan,
#             'diversity_h_max': np.nan,
#             'diversity_l2j_max': np.nan,
#             'diversity_mh_max': np.nan,
#             'count_diversity_cont': np.nan,
#             'count_diversity_cate': np.nan,
#             'count_diversity_all': np.nan,
#             'accuracy_knn_sklearn': 0.0,
#             'accuracy_knn_dist': 0.0,
#             'lof': np.nan,
#             'delta': 0.0,
#             'delta_min': 0.0,
#             'delta_max': 0.0,

#             'plausibility_sum': 0.0,
#             'plausibility_max_number_cf': 0.0,
#             'plausibility_number_cf': 0.0,
#             'plausibility_number_valid_cf': 0.0,
#             'plausibility_number_actionable_cf': 0.0,
#             'plausibility_number_valid_actionable_cf': 0.0,
#         }

#     return res


# columns = ['dataset',  'black_box', 'method', 'idx', 'k', 'known_train', 'search_diversity', 'metric',
#            'time_train', 'time_test', 'runtime', 'variable_features_flag',
#            'number_cf', 'number_valid_cf', 'perc_valid_cf', 'perc_valid_cf_all', 'number_actionable_cf', 'perc_actionable_cf',
#            'perc_actionable_cf_all', 'number_valid_actionable_cf', 'perc_valid_actionable_cf',
#            'perc_valid_actionable_cf_all', 'avg_number_violations_per_cf', 'avg_number_violations',
#            'distance_l2', 'distance_mad', 'distance_j', 'distance_h', 'distance_l2j', 'distance_mh',
#            'distance_l2_min', 'distance_mad_min', 'distance_j_min', 'distance_h_min', 'distance_l2j_min',
#            'distance_mh_min', 'distance_l2_max', 'distance_mad_max', 'distance_j_max', 'distance_h_max',
#            'distance_l2j_max', 'distance_mh_max', 'avg_number_changes_per_cf', 'avg_number_changes', 'diversity_l2',
#            'diversity_mad', 'diversity_j', 'diversity_h', 'diversity_l2j', 'diversity_mh', 'diversity_l2_min',
#            'diversity_mad_min', 'diversity_j_min', 'diversity_h_min', 'diversity_l2j_min', 'diversity_mh_min',
#            'diversity_l2_max', 'diversity_mad_max', 'diversity_j_max', 'diversity_h_max', 'diversity_l2j_max',
#            'diversity_mh_max', 'count_diversity_cont', 'count_diversity_cate', 'count_diversity_all',
#            'accuracy_knn_sklearn', 'accuracy_knn_dist', 'lof', 'delta', 'delta_min', 'delta_max', 'instability_si',
#            'plausibility_sum', 'plausibility_max_number_cf', 'plausibility_number_cf', 'plausibility_number_valid_cf',
#            'plausibility_number_actionable_cf', 'plausibility_number_valid_actionable_cf'
# ]





# def evaluate_only_plasubility(cf_list, x, classifier, y_val, max_number_cf, variable_features, continuous_features_all,
#                      categorical_features_all, X_train, X_test, ratio_cont):
#     number_cf_ = len(cf_list)
#     if number_cf_ > 0:

#         y_pred = classifier.predict(X_test)

#         number_valid_cf_ = number_valid_cf(cf_list, classifier, y_val)
#         perc_valid_cf_ = perc_valid_cf(cf_list, classifier, y_val, k=number_cf_)
#         perc_valid_cf_all_ = perc_valid_cf(cf_list, classifier, y_val, k=max_number_cf)
#         number_actionable_cf_ = number_actionable_cf(x, cf_list, variable_features)
#         perc_actionable_cf_ = perc_actionable_cf(x, cf_list, variable_features, k=number_cf_)
#         perc_actionable_cf_all_ = perc_actionable_cf(x, cf_list, variable_features, k=max_number_cf)
#         number_valid_actionable_cf_ = number_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features)
#         perc_valid_actionable_cf_ = perc_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features, k=number_cf_)
#         perc_valid_actionable_cf_all_ = perc_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features,
#                                                                  k=max_number_cf)

#         sum_dist = 0.0
#         for cf in cf_list:
#             y_cf_val = classifier.predict(cf.reshape(1, -1))[0]
#             X_test_y = X_test[y_cf_val == y_pred]
#             neigh_dist = distance_mh(x.reshape(1, -1), X_test_y, continuous_features_all,
#                                      categorical_features_all, X_train, ratio_cont)
#             idx_neigh = np.argsort(neigh_dist)[0]
#             closest = X_test_y[idx_neigh]
#             d = distance_mh(cf, closest.reshape(1, -1), continuous_features_all,
#                             categorical_features_all, X_train, ratio_cont)
#             sum_dist += d

#         plausibility_sum = sum_dist
#         plausibility_max_number_cf_ = sum_dist / max_number_cf
#         plausibility_number_cf_ = sum_dist / number_cf_
#         plausibility_number_valid_cf_ = sum_dist / number_valid_cf_ if number_valid_cf_ > 0 else sum_dist
#         plausibility_number_actionable_cf_ = sum_dist / number_actionable_cf_ if number_actionable_cf_ > 0 else sum_dist
#         plausibility_number_valid_actionable_cf_ = sum_dist / number_valid_actionable_cf_ if number_valid_actionable_cf_ > 0 else sum_dist
#     else:
#         number_valid_cf_ = 0.0
#         perc_valid_cf_ = 0.0
#         perc_valid_cf_all_ = 0.0
#         number_actionable_cf_ = 0.0
#         perc_actionable_cf_ = 0.0
#         perc_actionable_cf_all_ = 0.0
#         number_valid_actionable_cf_ = 0.0
#         perc_valid_actionable_cf_ = 0.0
#         perc_valid_actionable_cf_all_ = 0.0
#         plausibility_sum = 0.0
#         plausibility_max_number_cf_ = 0.0
#         plausibility_number_cf_ = 0.0
#         plausibility_number_valid_cf_ = 0.0
#         plausibility_number_actionable_cf_ = 0.0
#         plausibility_number_valid_actionable_cf_ = 0.0

#     res = {
#         'number_cf': number_cf_,
#         'number_valid_cf': number_valid_cf_,
#         'perc_valid_cf': perc_valid_cf_,
#         'perc_valid_cf_all': perc_valid_cf_all_,
#         'number_actionable_cf': number_actionable_cf_,
#         'perc_actionable_cf': perc_actionable_cf_,
#         'perc_actionable_cf_all': perc_actionable_cf_all_,
#         'number_valid_actionable_cf': number_valid_actionable_cf_,
#         'perc_valid_actionable_cf': perc_valid_actionable_cf_,
#         'perc_valid_actionable_cf_all': perc_valid_actionable_cf_all_,
#         'plausibility_sum': plausibility_sum,
#         'plausibility_max_number_cf': plausibility_max_number_cf_,
#         'plausibility_number_cf': plausibility_number_cf_,
#         'plausibility_number_valid_cf': plausibility_number_valid_cf_,
#         'plausibility_number_actionable_cf': plausibility_number_actionable_cf_,
#         'plausibility_number_valid_actionable_cf': plausibility_number_valid_actionable_cf_,
#     }

#     return res


# def plausibility_img_ts(x_true, x, classifier, cf_true, cf_list, X_test, y_pred, X_train, continuous_features):
#     sum_dist = 0.0
#     for i, cf in enumerate(cf_true):
#         y_cf_val = classifier.predict(cf.reshape((1,) + cf.shape))[0]
#         X_test_y = X_test[y_cf_val == y_pred]
#         neigh_dist = continuous_distance(x.reshape(1, -1), X_test_y, continuous_features, metric='mad', X=X_train)
#         idx_neigh = np.argsort(neigh_dist)[0]
#         closest = X_test_y[idx_neigh]
#         d = continuous_distance(cf_list[i], closest.reshape(1, -1), continuous_features, metric='mad', X=X_train)
#         sum_dist += d
#     return sum_dist


# def select_test_knn_img(x_true, x_flat, b, X_test, X_test_flat, test_size=5):
#     y_val = b.predict(x_true.reshape((1,) + x_true.shape))
#     y_test = b.predict(X_test)

#     dist_f = cdist(x_flat.reshape(1, -1), X_test_flat[y_test == y_val], metric='euclidean')
#     dist_cf = cdist(x_flat.reshape(1, -1), X_test_flat[y_test != y_val], metric='euclidean')

#     index_f = np.argsort(dist_f)[0][:test_size].tolist()
#     index_cf = np.argsort(dist_cf)[0][:test_size].tolist()

#     index = np.array(index_f + index_cf)

#     return X_test_flat[index], X_test[index]


# def accuracy_knn_sklearn_img_ts(x_true, x_flat, cf_list, b, X_test, X_test_flat, test_size=5):
#     clf = KNeighborsClassifier(n_neighbors=1)
#     X_train_true = np.vstack([x_true.reshape((1,) + x_true.shape), cf_list])
#     y_train = b.predict(X_train_true)
#     X_train_flat = np.array([x0.flatten() for x0 in X_train_true])
#     clf.fit(X_train_flat, y_train)

#     X_test_knn_flat, X_test_knn_true = select_test_knn_img(x_true, x_flat, b, X_test, X_test_flat, test_size)
#     y_test = b.predict(X_test_knn_true)
#     y_pred = clf.predict(X_test_knn_flat)

#     return accuracy_score(y_test, y_pred)


# def accuracy_knn_dist_img_ts(x_true, x_flat, cf_list_true, cf_list_flat, b, X_test_true, X_test_flat, test_size=5):
#     X_train_true = np.vstack([x_true.reshape((1,) + x_true.shape), cf_list_true])
#     X_train_flat = np.vstack([x_flat.reshape(1,-1), cf_list_flat])
#     y_train = b.predict(X_train_true)

#     X_test_knn_flat, X_test_knn_true = select_test_knn_img(x_true, x_flat, b, X_test_true, X_test_flat, test_size)
#     y_test = b.predict(X_test_knn_true)

#     y_pred = list()
#     for x_test in X_test_knn_flat:
#         dist = cdist(x_test.reshape(1, -1), X_train_flat, metric='euclidean')
#         idx = np.argmin(dist)
#         y_pred.append(y_train[idx])

#     return accuracy_score(y_test, y_pred)


# def evaluate_cf_list_img_ts(cf_list, x, classifier, y_val, max_number_cf, variable_features, X_train, X_test, number_features,
#                             X_train_flat, X_test_flat):

#     number_cf_ = len(cf_list)

#     if number_cf_ > 0:
#         scaler = DummyScaler()
#         scaler.fit(X_train)

#         y_pred = classifier.predict(X_test)

#         number_valid_cf_ = number_valid_cf(cf_list, classifier, y_val)
#         perc_valid_cf_ = perc_valid_cf(cf_list, classifier, y_val, k=number_cf_)
#         perc_valid_cf_all_ = perc_valid_cf(cf_list, classifier, y_val, k=max_number_cf)
#         # number_actionable_cf_ = number_actionable_cf(x, cf_list, variable_features)
#         # perc_actionable_cf_ = perc_actionable_cf(x, cf_list, variable_features, k=number_cf_)
#         # perc_actionable_cf_all_ = perc_actionable_cf(x, cf_list, variable_features, k=max_number_cf)
#         # number_valid_actionable_cf_ = number_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features)
#         # perc_valid_actionable_cf_ = perc_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features, k=number_cf_)
#         # perc_valid_actionable_cf_all_ = perc_valid_actionable_cf(x, cf_list, classifier, y_val, variable_features, k=max_number_cf)
#         x_flat = x.flatten()
#         cf_list_flat = np.array([cf.flatten() for cf in cf_list])
#         avg_number_violations_per_cf_ = avg_number_violations_per_cf(x_flat, cf_list_flat, variable_features)
#         avg_number_violations_ = avg_number_violations(x_flat, cf_list_flat, variable_features)

#         plausibility_sum = plausibility_img_ts(x, x_flat, classifier, cf_list, cf_list_flat, X_test_flat, y_pred, X_train_flat,
#                                                variable_features)
#         plausibility_max_number_cf_ = plausibility_sum / max_number_cf
#         plausibility_number_cf_ = plausibility_sum / number_cf_
#         plausibility_number_valid_cf_ = plausibility_sum / number_valid_cf_ if number_valid_cf_ > 0 else plausibility_sum
#         # plausibility_number_actionable_cf_ = plausibility_sum / number_actionable_cf_ if number_actionable_cf_ > 0 else plausibility_sum
#         # plausibility_number_valid_actionable_cf_ = plausibility_sum / number_valid_actionable_cf_ if number_valid_actionable_cf_ > 0 else plausibility_sum

#         distance_l2_ = continuous_distance(x_flat, cf_list_flat, variable_features, metric='euclidean', X=None)
#         distance_mad_ = continuous_distance(x_flat, cf_list_flat, variable_features, metric='mad', X=X_train_flat)

#         distance_l2_min_ = continuous_distance(x_flat, cf_list_flat, variable_features, metric='euclidean',
#                                                X=None, agg='min')
#         distance_mad_min_ = continuous_distance(x_flat, cf_list_flat, variable_features, metric='mad',
#                                                 X=X_train_flat, agg='min')

#         distance_l2_max_ = continuous_distance(x_flat, cf_list_flat, variable_features,
#                                                metric='euclidean', X=None, agg='max')
#         distance_mad_max_ = continuous_distance(x_flat, cf_list_flat, variable_features,
#                                                 metric='mad', X=X_train_flat, agg='max')

#         avg_number_changes_per_cf_ = avg_number_changes_per_cf(x_flat, cf_list_flat, variable_features)
#         avg_number_changes_ = avg_number_changes(x_flat, cf_list_flat, number_features, variable_features)

#         delta_ = delta_proba_img_ts(x, cf_list, classifier, agg='mean')
#         delta_min_ = delta_proba_img_ts(x, cf_list, classifier, agg='min')
#         delta_max_ = delta_proba_img_ts(x, cf_list, classifier, agg='max')

#         if len(cf_list) > 1:
#             diversity_l2_ = continuous_diversity(cf_list_flat, variable_features,
#                                                  metric='euclidean', X=None)
#             diversity_mad_ = continuous_diversity(cf_list_flat, variable_features, metric='mad',
#                                                   X=X_train_flat)
#             diversity_l2_min_ = continuous_diversity(cf_list_flat, variable_features,
#                                                      metric='euclidean', X=None, agg='min')
#             diversity_mad_min_ = continuous_diversity(cf_list_flat, variable_features,
#                                                       metric='mad', X=X_train_flat, agg='min')
#             diversity_l2_max_ = continuous_diversity(cf_list_flat, variable_features,
#                                                      metric='euclidean', X=None, agg='max')
#             diversity_mad_max_ = continuous_diversity(cf_list_flat, variable_features,
#                                                       metric='mad', X=X_train_flat, agg='max')

#         else:
#             diversity_l2_ = 0.0
#             diversity_mad_ = 0.0
#             diversity_l2_min_ = 0.0
#             diversity_mad_min_ = 0.0
#             diversity_l2_max_ = 0.0
#             diversity_mad_max_ = 0.0

#         count_diversity_ = count_diversity(cf_list_flat, variable_features, number_features, variable_features)

#         accuracy_knn_sklearn_ = accuracy_knn_sklearn_img_ts(x, x_flat, cf_list, classifier, X_test, X_test_flat, test_size=5)
#         accuracy_knn_dist_ = accuracy_knn_dist_img_ts(x, x_flat, cf_list, cf_list_flat, classifier, X_test, X_test_flat, test_size=5)

#         # lof_ = lof(x_flat, cf_list_flat, X_train_flat, scaler)

#         res = {
#             'number_cf': number_cf_,
#             'number_valid_cf': number_valid_cf_,
#             'perc_valid_cf': perc_valid_cf_,
#             'perc_valid_cf_all': perc_valid_cf_all_,
#             # 'number_actionable_cf': number_actionable_cf_,
#             # 'perc_actionable_cf': perc_actionable_cf_,
#             # 'perc_actionable_cf_all': perc_actionable_cf_all_,
#             # 'number_valid_actionable_cf': number_valid_actionable_cf_,
#             # 'perc_valid_actionable_cf': perc_valid_actionable_cf_,
#             # 'perc_valid_actionable_cf_all': perc_valid_actionable_cf_all_,
#             'avg_number_violations_per_cf': avg_number_violations_per_cf_,
#             'avg_number_violations': avg_number_violations_,
#             'distance_l2': distance_l2_,
#             'distance_mad': distance_mad_,
#             'avg_number_changes_per_cf': avg_number_changes_per_cf_,
#             'avg_number_changes': avg_number_changes_,

#             'distance_l2_min': distance_l2_min_,
#             'distance_mad_min': distance_mad_min_,
#             'distance_l2_max': distance_l2_max_,
#             'distance_mad_max': distance_mad_max_,
#             'diversity_l2': diversity_l2_,
#             'diversity_mad': diversity_mad_,

#             'diversity_l2_min': diversity_l2_min_,
#             'diversity_mad_min': diversity_mad_min_,
#             'diversity_l2_max': diversity_l2_max_,
#             'diversity_mad_max': diversity_mad_max_,

#             'count_diversity_': count_diversity_,
#             'accuracy_knn_sklearn': accuracy_knn_sklearn_,
#             'accuracy_knn_dist': accuracy_knn_dist_,
#             # 'lof': lof_,

#             'delta': delta_,
#             'delta_min': delta_min_,
#             'delta_max': delta_max_,

#             'plausibility_sum': plausibility_sum,
#             'plausibility_max_number_cf': plausibility_max_number_cf_,
#             'plausibility_number_cf': plausibility_number_cf_,
#             'plausibility_number_valid_cf': plausibility_number_valid_cf_,
#             # 'plausibility_number_actionable_cf': plausibility_number_actionable_cf_,
#             # 'plausibility_number_valid_actionable_cf': plausibility_number_valid_actionable_cf_,
#         }

#     else:
#         res = {
#             'number_cf': number_cf_,
#             'number_valid_cf': 0.0,
#             'perc_valid_cf': 0.0,
#             'perc_valid_cf_all': 0.0,
#             # 'number_actionable_cf': 0.0,
#             # 'perc_actionable_cf': 0.0,
#             # 'perc_actionable_cf_all': 0.0,
#             # 'number_valid_actionable_cf': 0.0,
#             # 'perc_valid_actionable_cf': 0.0,
#             # 'perc_valid_actionable_cf_all': 0.0,
#             'avg_number_violations_per_cf': np.nan,
#             'avg_number_violations': np.nan,
#             'distance_l2': np.nan,
#             'distance_mad': np.nan,
#             'avg_number_changes_per_cf': np.nan,
#             'avg_number_changes': np.nan,

#             'distance_l2_min': np.nan,
#             'distance_mad_min': np.nan,
#             'distance_l2_max': np.nan,
#             'distance_mad_max': np.nan,
#             'diversity_l2': np.nan,
#             'diversity_mad': np.nan,

#             'diversity_l2_min': np.nan,
#             'diversity_mad_min': np.nan,
#             'diversity_l2_max': np.nan,
#             'diversity_mad_max': np.nan,

#             'count_diversity_': np.nan,
#             'accuracy_knn_sklearn': 0.0,
#             'accuracy_knn_dist': 0.0,
#             # 'lof': np.nan,

#             'delta': 0.0,
#             'delta_min': 0.0,
#             'delta_max': 0.0,

#             'plausibility_sum': 0.0,
#             'plausibility_max_number_cf': 0.0,
#             'plausibility_number_cf': 0.0,
#             'plausibility_number_valid_cf': 0.0,
#             # 'plausibility_number_actionable_cf': 0.0,
#             # 'plausibility_number_valid_actionable_cf': 0.0,
#         }

#     return res


# columns_img_ts = ['dataset', 'black_box', 'method', 'idx', 'k',
#                'time_train', 'time_test', 'runtime',
#                'number_cf', 'number_valid_cf', 'perc_valid_cf', 'perc_valid_cf_all',
#                'avg_number_violations_per_cf', 'avg_number_violations',
#                'distance_l2', 'distance_mad', 'avg_number_changes_per_cf', 'avg_number_changes',
#                'distance_l2_min', 'distance_mad_min', 'distance_l2_max', 'distance_mad_max', 'diversity_l2',
#                'diversity_mad', 'diversity_l2_min', 'diversity_mad_min', 'diversity_l2_max', 'diversity_mad_max',
#                'count_diversity_', 'accuracy_knn_sklearn', 'accuracy_knn_dist', 'delta', 'delta_min', 'delta_max',
#                'plausibility_sum', 'plausibility_max_number_cf', 'plausibility_number_cf', 'plausibility_number_valid_cf',
# ]