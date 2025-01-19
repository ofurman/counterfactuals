import numpy as np

from .cet import CounterfactualExplanationTree


class GroupCF:
    def __init__(self, clf, feature_names, X_train, y_train, y_target):
        self.clf = clf
        self.y_target = y_target
        self.feature_names = feature_names

        feature_types = ["C"] * X_train.shape[1]
        feature_categories = []
        feature_constraints = [""] * X_train.shape[1]
        self.cf_tree = CounterfactualExplanationTree(
            self.clf,
            X_train,
            y_train,
            lime_approximation=True,
            feature_names=self.feature_names,
            feature_types=feature_types,
            feature_categories=feature_categories,
            feature_constraints=feature_constraints,
            target_name="y",
            target_labels=[0, 1],
        )

    def compute_explanation(self, X, LAMBDA=0.01, GAMMA=0.75):
        cet = self.cf_tree.fit(
            X,
            max_change_num=X.shape[1],
            cost_type="TLPS",
            C=LAMBDA,
            gamma=GAMMA,
            time_limit=1,
            verbose=False,
        )
        cet.print_tree()
        loss = cet.loss(X, target=self.y_target)
        delta_cf = cet.predict(X)

        return np.mean(delta_cf, axis=0), loss
