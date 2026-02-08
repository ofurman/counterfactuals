import logging
import warnings
from typing import Optional

import dice_ml
import numpy as np
import pandas as pd

# Suppress noisy warnings from downstream libraries during counterfactual generation.
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DiceExplainerWrapper:
    """Thin wrapper around DiCE to provide a simple generate interface."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        features_with_target: list[str],
        disc_model,
        desired_class: int = 1,
    ) -> None:
        self.features = features_with_target
        self.desired_class = desired_class

        input_dataframe = pd.DataFrame(
            np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
            columns=self.features,
        )

        dice = dice_ml.Data(
            dataframe=input_dataframe,
            continuous_features=self.features[:-1],
            outcome_name=self.features[-1],
        )
        model = dice_ml.Model(disc_model, backend="PYT")
        self.exp = dice_ml.Dice(dice, model)

    def generate(
        self,
        query_instance: pd.DataFrame,
        desired_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Generate a single counterfactual for the given query instance."""
        desired = self.desired_class if desired_class is None else desired_class
        query_instance = pd.DataFrame(query_instance, columns=self.features[:-1])
        try:
            dice_exp = self.exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class=desired,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            return None

        if dice_exp.cf_examples_list[0].final_cfs_df is not None:
            return self.get_counterfactual(dice_exp)
        return None

    def get_counterfactual(self, dice_exp) -> np.ndarray:
        """Extract counterfactual array without the target column."""
        return dice_exp.cf_examples_list[0].final_cfs_df.to_numpy()[:, :-1]
