from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .mlmodel import MLModel


def check_counterfactuals(
    mlmodel: MLModel,
    counterfactuals: Union[List, pd.DataFrame],
    factuals_index: pd.Index,
    negative_label: int = 0,
) -> pd.DataFrame:
    """
    Takes the generated list of counterfactuals from recourse methods and checks if these samples are able
    to flip the label from 0 to 1. Every counterfactual which still has a negative label, will be replaced with an
    empty row.

    Parameters
    ----------
    mlmodel:
        Black-box-model we want to discover.
    counterfactuals:
        List or DataFrame of generated samples from recourse method.
    factuals_index:
        Index of the original factuals DataFrame.
    negative_label:
        Defines the negative label.

    Returns
    -------
    pd.DataFrame
    """

    if isinstance(counterfactuals, list):
        df_cfs = pd.DataFrame(
            np.array(counterfactuals),
            columns=mlmodel.feature_input_order,
            index=factuals_index.copy(),
        )
    else:
        df_cfs = counterfactuals.copy()

    df_cfs[mlmodel.data.target] = np.argmax(mlmodel.predict_proba(df_cfs), axis=1)
    # Change all wrong counterfactuals to nan
    df_cfs.loc[df_cfs[mlmodel.data.target] == negative_label, :] = np.nan

    return df_cfs


def merge_default_parameters(hyperparams: Optional[Dict], default: Dict) -> Dict:
    """
    Checks if the input parameter hyperparams contains every necessary key and if not, uses default values or
    raises a ValueError if no default value is given.

    Parameters
    ----------
    hyperparams: dict
        Hyperparameter as passed to the recourse method.
    default: dict
        Dictionary with every necessary key and default value.
        If key has no default value and hyperparams has no value for it, raise a ValueError

    Returns
    -------
    dict
        Dictionary with every necessary key.
    """
    if hyperparams is None:
        return default

    keys = default.keys()
    dict_output = dict()

    for key in keys:
        if isinstance(default[key], dict):
            hyperparams[key] = dict() if key not in hyperparams.keys() else hyperparams[key]
            sub_dict = merge_default_parameters(hyperparams[key], default[key])
            dict_output[key] = sub_dict
            continue
        if key not in hyperparams.keys():
            default_val = default[key]
            if default_val is None:
                # None value for key depicts that user has to pass this value in hyperparams
                raise ValueError(
                    "For {} is no default value defined, please pass this key and its value in hyperparams".format(
                        key
                    )
                )
            elif isinstance(default_val, str) and default_val == "_optional_":
                # _optional_ depicts that value for this key is optional and therefore None
                default_val = None
            dict_output[key] = default_val
        else:
            if hyperparams[key] is None:
                raise ValueError("For {} in hyperparams is a value needed".format(key))
            dict_output[key] = hyperparams[key]

    return dict_output


def reconstruct_encoding_constraints(
    x: torch.Tensor, feature_groups: List[List[int]]
) -> torch.Tensor:
    """
    Project one-hot-encoded categorical blocks back onto valid vertices, so that
    each categorical group has exactly one active column (sum == 1) and no
    contradictions (e.g., sex_female = 1 and sex_male = 1 at the same time).

    The decoder produces soft values per column; rounding each column
    independently can yield multiple active columns or an all-zero group. Taking
    the argmax within each group guarantees a valid one-hot vector.

    Parameters
    ----------
    x:
        Instance(s) whose categorical blocks should be made one-hot-valid.
    feature_groups:
        List of column-index groups, one per original categorical feature (as
        produced by ``categorical_features_lists``). A single-column group is
        treated as a binary indicator and simply rounded.

    Returns
    -------
    Tensor with each categorical group snapped to a valid one-hot vector.
    """
    x_enc = x.clone()

    for group in feature_groups:
        if not group:
            continue
        if len(group) == 1:
            x_enc[:, group[0]] = torch.round(x_enc[:, group[0]])
            continue

        idx = torch.as_tensor(group, device=x_enc.device, dtype=torch.long)
        winner = torch.argmax(x_enc[:, idx], dim=1)
        one_hot = torch.zeros((x_enc.shape[0], len(group)), dtype=x_enc.dtype, device=x_enc.device)
        one_hot[torch.arange(x_enc.shape[0]), winner] = 1.0
        x_enc[:, idx] = one_hot

    return x_enc
