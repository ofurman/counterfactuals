import dice_ml
import numpy as np
import pandas as pd


def compute_cf(clf, x, y_target, X_train, y_train, n_cf=1, verbose=True):
    try:
        # Wrapping
        cols = [f"{i}" for i in range(X_train.shape[1])]
        X_df = pd.DataFrame(X_train, columns=cols)
        x_orig = pd.DataFrame(x.reshape(1, -1), columns=cols)
        y_df = pd.DataFrame(y_train, columns=["y"]).astype(np.int32)
        data_df = pd.concat([X_df, y_df], axis=1)

        data = dice_ml.Data(
            dataframe=data_df, continuous_features=cols, outcome_name="y"
        )
        model = dice_ml.Model(model=clf, backend="sklearn")

        # Compute counterfactual by using a genetic algorithm
        cf_algo = dice_ml.Dice(data, model, method="genetic")
        cf_result = cf_algo.generate_counterfactuals(
            x_orig, total_CFs=n_cf, desired_class="opposite", verbose=False
        )

        X_cf = cf_result.cf_examples_list[0].final_cfs_df[cols].to_numpy()
        Y_cf = cf_result.cf_examples_list[0].final_cfs_df["y"].to_numpy().flatten()

        for i in range(n_cf):
            if Y_cf[i] != y_target:
                return None

        return [X_cf[i, :].flatten() for i in range(X_cf.shape[0])]
    except Exception as ex:
        if verbose is True:
            print(ex)
        return None


class DiceExplainer:
    def __init__(self, clf, X_train, y_train):
        self.clf = clf
        self.X_train = X_train
        self.y_train = y_train

    def compute_counterfactual(self, x, y_target):
        return compute_cf(self.clf, x, y_target, self.X_train, self.y_train)
