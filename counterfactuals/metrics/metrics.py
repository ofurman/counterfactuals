import torch
import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import _validate_vector, cdist


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
    val = torch.sum(valid_cf(y, y_cf))
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
    val = np.sum(val.numpy())
    return val / (number_cf * number_features)


def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def continuous_distance(
    X, X_cf, continuous_features, metric="euclidean", X_all=None, agg="mean", _diag=True
):
    assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
    if metric == "mad":
        mad = median_abs_deviation(X_all[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)

        metric = _mad_cityblock

    dist = cdist(X[:, continuous_features], X_cf[:, continuous_features], metric=metric)

    dist = np.diag(dist) if _diag else dist
    agg_funcs = {"mean": np.mean, "max": np.max, "min": np.min, "no": lambda x: x}
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    return agg_funcs[agg](dist)


def categorical_distance(
    X, X_cf, categorical_features, metric="jaccard", agg=None, _diag=True
):
    assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
    dist = cdist(
        X[:, categorical_features], X_cf[:, categorical_features], metric=metric
    )
    dist = np.diag(dist) if _diag else dist
    agg_funcs = {"mean": np.mean, "max": np.max, "min": np.min, "no": lambda x: x}
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    return agg_funcs[agg](dist)


def distance_l2_jaccard(
    X, X_cf, continuous_features, categorical_features, ratio_cont=None
):
    number_features = X_cf.shape[1]
    dist_cont = continuous_distance(
        X, X_cf, continuous_features, metric="euclidean", X_all=None, agg="mean"
    )
    dist_cate = categorical_distance(
        X, X_cf, categorical_features, metric="jaccard", agg="mean"
    )
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / number_features
        ratio_categorical = len(categorical_features) / number_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_continuous
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def distance_mad_hamming(
    X,
    X_cf,
    continuous_features,
    categorical_features,
    X_all,
    ratio_cont=None,
    agg=None,
    diag=True,
):
    number_features = X_cf.shape[1]
    dist_cont = continuous_distance(
        X, X_cf, continuous_features, metric="mad", X_all=X_all, agg=agg, _diag=diag
    )
    dist_cate = categorical_distance(
        X, X_cf, categorical_features, metric="hamming", agg=agg, _diag=diag
    )
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / number_features
        ratio_categorical = len(categorical_features) / number_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def number_changes_per_cf(X, X_cf, continuous_features, agg="mean"):
    assert X.shape == X_cf.shape, f"Shapes should be the same: {X.shape} - {X_cf.shape}"
    result = np.sum(X[:, continuous_features] == X_cf[:, continuous_features], axis=1)
    agg_funcs = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
        "no": lambda x: x,
    }
    assert agg in agg_funcs.keys(), f"Param agg should be one of: {agg_funcs.keys()}"
    return agg_funcs[agg](result)


def avg_number_changes(X, X_cf, continuous_features):
    number_cf, number_features = X_cf.shape[1]
    val = number_changes_per_cf(X, X_cf, continuous_features, agg="sum")
    return val / (number_cf * number_features)


def plausibility(
    X_test,
    X_cf,
    y_test,
    continuous_features_all,
    categorical_features_all,
    X_train,
    ratio_cont=None,
):
    dist_neighb = distance_mad_hamming(
        X_test,
        X_test,
        continuous_features_all,
        categorical_features_all,
        X_train,
        ratio_cont=ratio_cont,
        agg="no",
        diag=False,
    )
    dist_neighb[y_test == 0, y_test == 0] = np.inf
    dist_neighb[y_test == 1, y_test == 1] = np.inf
    idx_neighb = np.argmin(dist_neighb, axis=0)
    dist_neighb = distance_mad_hamming(
        X_test[idx_neighb],
        X_cf,
        continuous_features_all,
        categorical_features_all,
        X_train,
        ratio_cont=ratio_cont,
        agg="mean",
        diag=True,
    )
    return dist_neighb


def delta_proba(x, cf_list, classifier, agg=None):
    y_val = classifier.predict_proba(x)
    y_cf = classifier.predict_proba(cf_list)
    deltas = np.abs(y_cf - y_val)

    if agg is None or agg == "mean":
        return np.mean(deltas)

    if agg == "max":
        return np.max(deltas)

    if agg == "min":
        return np.min(deltas)


def calc_gen_model_density(gen_log_probs_cf, gen_log_probs_xs, ys):
    log_density_cfs = []
    log_density_xs = []
    for y in np.unique(ys):
        log_density_cfs.append(gen_log_probs_cf[y.astype(int), ys != y])
        log_density_xs.append(gen_log_probs_xs[y.astype(int), ys == y])
    return np.mean(np.hstack(log_density_cfs)), np.mean(np.hstack(log_density_xs))


def local_outlier_factor(X_train, X, X_cf, n_neighbors=20):
    """
    Calculate the Local Outlier Factor (LOF) for each sample in X and X_cf.
    LOF(k) ~ 1 means Similar density as neighbors,

    LOF(k) < 1 means Higher density than neighbors (Inlier),

    LOF(k) > 1 means Lower density than neighbors (Outlier)
    """
    X_all = np.concatenate([X_train, X, X_cf], axis=0)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit(X_all)
    lof_scores = -lof.negative_outlier_factor_
    return lof_scores[: len(X)], lof_scores[len(X) :]


def isolation_forest_metric(X_train, X, X_cf, n_estimators=100):
    """
    Calculate the Isolation Forest score for each sample in X and X_cf.
    The score is between -0.5 and 0.5, where smaller values mean more anomalous.
    https://stackoverflow.com/questions/45223921/what-is-the-range-of-scikit-learns-isolationforest-decision-function-scores#51882974
    """
    X_all = np.concatenate([X, X_cf], axis=0)
    clf = IsolationForest(n_estimators=n_estimators)
    clf.fit(X_train)
    scores = clf.score_samples(X_all)
    return scores[: len(X)], scores[len(X) :]


def evaluate_cf(
    disc_model: torch.nn.Module,
    gen_model: torch.nn.Module,
    X_cf: np.ndarray,
    y_target: np.ndarray,
    model_returned: np.ndarray,
    continuous_features: list,
    categorical_features: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    delta: np.ndarray,
):
    assert isinstance(X_cf, np.ndarray)
    assert X_cf.dtype == np.float32

    X_test = X_test[model_returned]
    y_test = y_test[model_returned]
    y_target = y_target[model_returned]

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    X_cf = torch.from_numpy(X_cf)

    y_train = torch.from_numpy(y_train.reshape(-1))
    y_test = torch.from_numpy(y_test.reshape(-1))
    y_target = torch.from_numpy(y_target.reshape(-1))

    # Define variables for metrics
    ys_cfs_disc_pred = disc_model.predict(X_cf)
    model_returned_smth = np.sum(model_returned) / len(model_returned)
    valid_cf_disc_metric = perc_valid_cf(y_test, y_cf=ys_cfs_disc_pred)

    if X_test.shape[1:] != X_cf.shape[1:]:
        return dict(
            valid_cf_disc_metric=valid_cf_disc_metric,
            model_returned_smth=model_returned_smth,
        )

    # labels = np.unique(y_train).astype(np.float32)

    lof_scores_xs, lof_scores_cfs = local_outlier_factor(X_train, X_test, X_cf)
    isolation_forest_scores_xs, isolation_forest_scores_cfs = isolation_forest_metric(
        X_train, X_test, X_cf
    )

    # gen_log_probs_xs_all_classes = np.vstack([
    #     gen_model(torch.from_numpy(X_test), torch.full((X_test.shape[0], 1), target_label)).detach().numpy()
    #     for target_label in labels
    # ])
    # gen_log_probs_cf_all_classes = np.vstack([
    #     gen_model(torch.from_numpy(X_cf), torch.full((X_cf.shape[0], 1), target_label)).detach().numpy()
    #     for target_label in labels
    # ])

    gen_log_probs_xs = gen_model(X_test, y_test.type(torch.float32))
    gen_log_probs_cf = gen_model(X_cf, y_target.type(torch.float32))
    flow_prob_condition_acc = torch.sum(delta < gen_log_probs_cf) / len(
        gen_log_probs_cf
    )
    # ys_disc_pred = np.array(disc_model.predict(X))
    ys_cfs_disc_pred = disc_model.predict(X_cf)

    hamming_distance_metric = categorical_distance(
        X=X_test,
        X_cf=X_cf,
        categorical_features=categorical_features,
        metric="hamming",
        agg="mean",
    )
    jaccard_distance_metric = categorical_distance(
        X=X_test,
        X_cf=X_cf,
        categorical_features=categorical_features,
        metric="jaccard",
        agg="mean",
    )
    manhattan_distance_metric = continuous_distance(
        X=X_test,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="cityblock",
        X_all=X_test,
    )
    euclidean_distance_metric = continuous_distance(
        X=X_test,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="euclidean",
        X_all=X_test,
    )
    mad_distance_metric = continuous_distance(
        X=X_test,
        X_cf=X_cf,
        continuous_features=continuous_features,
        metric="mad",
        X_all=X_test,
    )
    l2_jaccard_distance_metric = distance_l2_jaccard(
        X=X_test,
        X_cf=X_cf,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
    mad_hamming_distance_metric = distance_mad_hamming(
        X=X_test,
        X_cf=X_cf,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        X_all=X_test,
        agg="mean",
    )
    sparsity_metric = sparsity(X_test, X_cf)

    # Create a dictionary of metrics
    metrics = {
        "model_returned_smth": model_returned_smth,
        "valid_cf_disc": valid_cf_disc_metric.item(),
        "dissimilarity_proximity_categorical_hamming": hamming_distance_metric,
        "dissimilarity_proximity_categorical_jaccard": jaccard_distance_metric,
        "dissimilarity_proximity_continuous_manhatan": manhattan_distance_metric,
        "dissimilarity_proximity_continuous_euclidean": euclidean_distance_metric,
        "dissimilarity_proximity_continuous_mad": mad_distance_metric,
        "distance_l2_jaccard": l2_jaccard_distance_metric,
        "distance_mad_hamming": mad_hamming_distance_metric,
        "sparsity": sparsity_metric,
    }

    metrics.update(
        {
            "flow_log_density_cfs": gen_log_probs_cf.mean().item(),
            "flow_log_density_xs": gen_log_probs_xs.mean().item(),
            "flow_prob_condition_acc": flow_prob_condition_acc.item(),
            "lof_scores_xs": lof_scores_xs.mean(),
            "lof_scores_cfs": lof_scores_cfs.mean(),
            "isolation_forest_scores_xs": isolation_forest_scores_xs.mean(),
            "isolation_forest_scores_cfs": isolation_forest_scores_cfs.mean(),
        }
    )
    return metrics
