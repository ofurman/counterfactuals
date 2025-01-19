import numpy as np
from sklearn.cluster import DBSCAN


def cluster_instances(X_samples, X_cf_samples, method="dbscan-cf"):
    if method == "dbscan-cf":
        try:
            clustering = DBSCAN(min_samples=2, eps=0.3, metric="cosine").fit(
                X_cf_samples
            )
            if (
                len(np.unique(clustering.labels_)) > 2
                and len(np.unique(clustering.labels_)) < 15
            ):
                return clustering
        except Exception as ex:
            print(ex)
    elif method == "dbscan-xorig":
        try:
            clustering = DBSCAN(eps=3, min_samples=2, metric="euclidean").fit(X_samples)
            if (
                len(np.unique(clustering.labels_)) > 2
                and len(np.unique(clustering.labels_)) < 10
            ):
                return clustering
        except Exception as ex:
            print(ex)
