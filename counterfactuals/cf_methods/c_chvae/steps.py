from typing import List

import pandas as pd


def order_data(feature_order: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Restores the correct input feature order for the ML model

    Only works for encoded data

    Parameters
    ----------
    feature_order : list
        List of input feature in correct order
    df : pd.DataFrame
        Data we want to order

    Returns
    -------
    output : pd.DataFrame
        Whole DataFrame with ordered feature
    """
    return df[feature_order]
