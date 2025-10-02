import logging

import dice_ml
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DiceExplainerWrapper:
    def __init__(self, X_train, y_train, features, disc_model):
        self.features = features
        input_dataframe = pd.DataFrame(
            np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
            columns=features,
        )

        dice = dice_ml.Data(
            dataframe=input_dataframe,
            continuous_features=features[:-1],
            outcome_name=features[-1],
        )
        model = dice_ml.Model(disc_model, backend="PYT")

        self.exp = dice_ml.Dice(dice, model, method="gradient")

    def generate(self, query_instance):
        query_instance = pd.DataFrame(query_instance, columns=self.features[:-1])
        dice_exp = self.exp.generate_counterfactuals(
            query_instance, total_CFs=1, desired_class="opposite"
        )
        if dice_exp.cf_examples_list[0].final_cfs_df is not None:
            counterfactual = self.get_counterfactual(dice_exp)
            return counterfactual

    def get_counterfactual(self, dice_exp):
        return dice_exp.cf_examples_list[0].final_cfs_df.to_numpy()[:, :-1]
