import numpy as np
from sklearn.cluster import DBSCAN


def cluster_instances(X_samples, X_cf_samples, method="dbscan-cf"):
    if method == "dbscan-cf":
        try:
            clustering = DBSCAN(min_samples=2, eps=0.1, metric="cosine").fit(
                X_cf_samples
            )
            return clustering
            if (
                len(np.unique(clustering.labels_)) > 2
                and len(np.unique(clustering.labels_)) < 15
            ):
                return clustering
            else:
                raise Exception(
                    f"Number of clusters is {len(np.unique(clustering.labels_))}"
                )
        except Exception as ex:
            print(ex)
    elif method == "dbscan-xorig":
        try:
            # for eps in [0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001]:
            for eps in [3.0, 2.0, 1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]:
                print(f"eps: {eps}")
                clustering = DBSCAN(eps=eps, min_samples=5, metric="euclidean").fit(
                    X_samples
                )
                print(f"Number of clusters: {len(np.unique(clustering.labels_))}")
                if (
                    len(np.unique(clustering.labels_)) > 2
                    and len(np.unique(clustering.labels_)) < 10
                ):
                    break
            print(f"Number of clusters: {len(np.unique(clustering.labels_))}")
            return clustering
        except Exception as ex:
            print(ex)
