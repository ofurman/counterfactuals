import numpy as np

from .cf_dice import DiceExplainer


class GroupCounterfactual:
    def __init__(self, clf, X_train, y_train):
        self.clf = clf
        self.cf_algo = DiceExplainer(clf, X_train, y_train)

    def __compute_individual_counterfactual(self, x, y_target):
        return self.cf_algo.compute_counterfactual(x, y_target)

    def compute_explanation(
        self, X_samples, y_target, X_contrasting_samples, prob_threshold=0.7
    ):
        delta_cf = np.zeros(X_samples.shape[1])

        # Compute individual counterfactuals
        Delta_cf = []
        for x in X_samples:
            x_cf = self.__compute_individual_counterfactual(x, y_target)
            if x_cf is None:
                continue
            deltacf = x_cf - x
            Delta_cf.append(deltacf)

        # Find most common feature subset
        features_idx = []
        for deltacf in Delta_cf:
            delta_idx = np.argwhere(deltacf != 0).flatten().tolist()
            features_idx += delta_idx
        f_ids, p = np.unique(features_idx, return_counts=True)
        p = p / (1.0 * np.sum(p))

        p_idx = np.argsort(p)[::-1]
        p_total = 0.0
        features_important_idx = []
        for pi in p_idx:
            features_important_idx.append(f_ids[pi])
            p_total += p[pi]
            if p_total >= prob_threshold:
                break

        # Compute group counterfactual by sampling feature values from instances in the contrasting class and use those as substitutes (i.e. no delta is computed, everything is treated as a categorical variable!)
        cur_best_coverage = 0.0
        for xcf in X_contrasting_samples:
            delta_cf_ = np.zeros(X_samples.shape[1])
            for f_id in features_important_idx:
                delta_cf_[f_id] = xcf[f_id]

            # Check coverage (i.e. feasibility)
            X_samples_ = np.copy(X_samples)
            coverage = []
            for x in X_samples_:
                for f_id in features_important_idx:
                    x[f_id] = delta_cf_[f_id]
                    if self.clf.predict([x]) == y_target:
                        coverage.append(1)
                    else:
                        coverage.append(0)
            cov_final = np.sum(coverage) / len(coverage)
            if cov_final >= cur_best_coverage:
                cur_best_coverage = cov_final
                delta_cf = delta_cf_

        return delta_cf, cur_best_coverage


def compute_groupcf(
    clf,
    X_train,
    y_train,
    X_samples,
    y_target,
    X_contrasting_samples,
    prob_threshold=0.7,
):
    expl = GroupCounterfactual(clf, X_train, y_train)
    return expl.compute_explanation(
        X_samples, y_target, X_contrasting_samples, prob_threshold
    )
