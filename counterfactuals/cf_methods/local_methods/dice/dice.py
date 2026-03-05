import logging

import dice_ml
import numpy as np
import pandas as pd

from counterfactuals.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from counterfactuals.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)

logger = logging.getLogger(__name__)


class DICE(BaseCounterfactualMethod, LocalCounterfactualMixin):
    """An interface class to different DiCE implementations."""

    def __init__(self, X_train, y_train, features, disc_model):
        self.features = features
        self.target_feature = "target"
        input_dataframe = pd.DataFrame(
            np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
            columns=features + [self.target_feature],
        )

        dice = dice_ml.Data(
            dataframe=input_dataframe,
            continuous_features=features[:-1],
            outcome_name=features[-1],
        )
        model = dice_ml.Model(disc_model, backend="PYT")

        self.exp = dice_ml.Dice(dice, model, method="gradient")

    def explain(
        self,
        Xs,
        ys,
        total_CFs=1,
        desired_class="opposite",
        desired_range=None,
        permitted_range=None,
        features_to_vary="all",
        stopping_threshold=0.5,
        posthoc_sparsity_param=0.1,
        posthoc_sparsity_algorithm="linear",
        verbose=False,
        **kwargs,
    ):
        """General method for generating counterfactuals.

        :param query_instances: Input point(s) for which counterfactuals are to be generated.
                                This can be a dataframe with one or more rows.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value
                              is "opposite" to the outcome class of query_instance for binary classification.
        :param desired_range: For regression problems. Contains the outcome range to
                              generate counterfactuals in. This should be a list of two numbers in
                              ascending order.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values.
                                Defaults to the range inferred from training data.
                                If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance. Used by ['genetic', 'gradientdescent'],
                                 ignored by ['random', 'kdtree'] methods.
        :param sparsity_weight: A positive float. Larger this weight, less features are changed from the query_instance.
                                Used by ['genetic', 'kdtree'], ignored by ['random', 'gradientdescent'] methods.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
                                 Used by ['genetic', 'gradientdescent'], ignored by ['random', 'kdtree'] methods.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
                                 Used by ['genetic', 'gradientdescent'], ignored by ['random', 'kdtree'] methods.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large (for instance,
                                           income varying from 10k to 1000k) and only if the features share a
                                           monotonic relationship with predicted outcome in the model.
        :param verbose: Whether to output detailed messages.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        :param kwargs: Other parameters accepted by specific explanation method

        :returns: A CounterfactualExplanations object that contains the list of
                  counterfactual examples per query_instance as one of its attributes.
        """
        query_instances = pd.DataFrame(Xs, columns=self.features)
        dice_exp = self.exp.generate_counterfactuals(
            query_instances,
            total_CFs,
            desired_class=desired_class,
            desired_range=desired_range,
            permitted_range=permitted_range,
            features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold,
            posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            verbose=verbose,
            **kwargs,
        )
        coverage_mask = np.array([cf.final_cfs_df.shape[0] > 0 for cf in dice_exp.cf_examples_list])
        Xs_cfs = self.get_counterfactual(dice_exp)
        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs = np.array(Xs)[coverage_mask]
        ys = np.array(ys)[coverage_mask]
        ys_target = [desired_class] * len(Xs) if desired_class != "opposite" else np.abs(1 - ys)
        return ExplanationResult(x_cfs=Xs_cfs, y_cf_targets=ys_target, x_origs=Xs, y_origs=ys)

    def explain_dataloader(
        self,
        dataloader,
        total_CFs=1,
        desired_class="opposite",
        desired_range=None,
        permitted_range=None,
        features_to_vary="all",
        stopping_threshold=0.5,
        posthoc_sparsity_param=0.1,
        posthoc_sparsity_algorithm="linear",
        verbose=False,
        **kwargs,
    ):
        """General method for generating counterfactuals.

        :param query_instances: Input point(s) for which counterfactuals are to be generated.
                                This can be a dataframe with one or more rows.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value
                              is "opposite" to the outcome class of query_instance for binary classification.
        :param desired_range: For regression problems. Contains the outcome range to
                              generate counterfactuals in. This should be a list of two numbers in
                              ascending order.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values.
                                Defaults to the range inferred from training data.
                                If None, uses the parameters initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance. Used by ['genetic', 'gradientdescent'],
                                 ignored by ['random', 'kdtree'] methods.
        :param sparsity_weight: A positive float. Larger this weight, less features are changed from the query_instance.
                                Used by ['genetic', 'kdtree'], ignored by ['random', 'gradientdescent'] methods.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
                                 Used by ['genetic', 'gradientdescent'], ignored by ['random', 'kdtree'] methods.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
                                 Used by ['genetic', 'gradientdescent'], ignored by ['random', 'kdtree'] methods.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large (for instance,
                                           income varying from 10k to 1000k) and only if the features share a
                                           monotonic relationship with predicted outcome in the model.
        :param verbose: Whether to output detailed messages.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        :param kwargs: Other parameters accepted by specific explanation method

        :returns: A CounterfactualExplanations object that contains the list of
                  counterfactual examples per query_instance as one of its attributes.
        """
        Xs, ys = dataloader.dataset.tensors
        Xs = Xs.numpy()[:5]
        ys = ys.numpy()[:5]
        Xs = pd.DataFrame(Xs, columns=self.features)
        dice_exp = self.exp.generate_counterfactuals(
            Xs,
            total_CFs=total_CFs,
            desired_class=desired_class,
            desired_range=None,
            permitted_range=permitted_range,
            features_to_vary=features_to_vary,
            stopping_threshold=stopping_threshold,
            posthoc_sparsity_param=posthoc_sparsity_param,
            posthoc_sparsity_algorithm=posthoc_sparsity_algorithm,
            verbose=verbose,
            **kwargs,
        )
        coverage_mask = np.array([cf.final_cfs_df.shape[0] > 0 for cf in dice_exp.cf_examples_list])
        Xs_cfs = self.get_counterfactual(dice_exp)
        Xs_cfs = np.array(Xs_cfs).squeeze()
        Xs = np.array(Xs)[coverage_mask]
        ys = np.array(ys)[coverage_mask]
        ys_target = [desired_class] * len(Xs) if desired_class != "opposite" else np.abs(1 - ys)
        return ExplanationResult(x_cfs=Xs_cfs, y_cf_targets=ys_target, x_origs=Xs, y_origs=ys)

    def get_counterfactual(self, dice_exp):
        return [
            cf.final_cfs_df[self.features].to_numpy()
            for cf in dice_exp.cf_examples_list
            if cf.final_cfs_df.shape[0] > 0
        ]
