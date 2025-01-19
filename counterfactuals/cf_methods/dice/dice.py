import numpy as np
import pandas as pd
import dice_ml
from tqdm import tqdm
from torch.utils.data import DataLoader

from counterfactuals.cf_methods.base import BaseCounterfactual, ExplanationResult
from counterfactuals.discriminative_models.base import BaseDiscModel


class DiCE(BaseCounterfactual):
    def __init__(
        self,
        disc_model: BaseDiscModel,
        train_dataset: pd.DataFrame,  # should be train dataset with target as a last column
        target_class: int = "opposite",  # any class other than origin will do
        **kwargs,  # ignore other arguments
    ) -> None:
        self.target_class = target_class
        self.dice_data = dice_ml.Data(
            dataframe=train_dataset,
            continuous_features=train_dataset.columns[:-1],
            outcome_name=train_dataset.columns[-1],
        )
        self.dice_model = dice_ml.Model(model=disc_model, backend="PYT")
        self.dice_exp = dice_ml.Dice(self.dice_data, self.dice_model)

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        try:
            dice_exp = self.dice_exp.generate_counterfactuals(
                X, total_CFs=1, desired_class=self.target_class, verbose=False
            )
            explanation = dice_exp.cf_examples_list[0].final_cfs_df.to_numpy()[:, :-1]
        except Exception as e:
            explanation = None
            print(e)
        return explanation, X, y_origin, y_target
        # return ExplanationResult(
        #     x_cfs=explanation, y_cf_targets=y_target, x_origs=X, y_origs=y_origin
        # )

    def explain_dataloader(
        self, dataloader: DataLoader, target_class: int, *args, **kwargs
    ) -> ExplanationResult:
        Xs, ys = dataloader.dataset.tensors
        # create ys_target numpy array same shape as ys but with target class
        # ys_target = np.full(ys.shape, target_class)
        ys_target = np.zeros_like(ys)
        ys_target[:, target_class] = 1
        Xs_cfs = []
        model_returned = []
        for X, y in tqdm(zip(Xs, ys), total=len(Xs)):
            try:
                dice_exp = self.dice_exp.generate_counterfactuals(
                    X, total_CFs=1, desired_class=self.target_class, verbose=False
                )
                explanation = dice_exp.cf_examples_list[0].final_cfs_df.to_numpy()[
                    :, :-1
                ]
                model_returned.append(True)
            except Exception as e:
                explanation = [[np.nan] * X.shape[1]]
                print(e)
                model_returned.append(False)
            Xs_cfs.append(explanation)

        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs = np.array(Xs)
        ys = np.array(ys)
        ys_target = np.array(ys_target)
        return Xs_cfs, Xs, ys, ys_target, model_returned
        # return ExplanationResult(x_cfs=Xs_cfs, y_cf_targets=ys, x_origs=Xs, y_origs=ys)
