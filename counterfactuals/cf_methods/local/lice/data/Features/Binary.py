from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from counterfactuals.cf_methods.local.lice.data.Features.Feature import (
    Feature,
    Monotonicity,
)
from counterfactuals.cf_methods.local.lice.data.Types import CategValue, OneDimData


class Binary(Feature):
    def __init__(
        self,
        training_vals: OneDimData,
        value_names: Optional[list[CategValue]] = None,
        name: Optional[str] = None,
        monotone: Monotonicity = Monotonicity.NONE,
        modifiable: bool = True,
    ):
        super().__init__(training_vals, name, monotone, modifiable)
        if value_names is None:
            value_names = np.unique(training_vals)
        else:
            valid_vals = np.isin(training_vals, value_names)
            if not np.all(valid_vals):
                raise ValueError(
                    f"""Incorrect value in a binary feature {self.name}.
                    Values {np.unique(training_vals[~valid_vals])} are not one of {value_names}"""
                )
        self.__negative_val, self.__positive_val = value_names
        self._MAD = np.asarray(
            [1.48 * np.nanstd(self.encode(training_vals, one_hot=False))]
        )

    @Feature._check_dims_on_encode
    def encode(
        self, vals: OneDimData, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        positive = vals == self.__positive_val
        if np.any(vals[~positive] != self.__negative_val):
            unknown = vals[~positive] != self.__negative_val
            raise ValueError(
                f"""Incorrect value in a binary feature {self.name}.
                Values {vals[~positive][unknown]} are not one of [{self.__negative_val},{self.__positive_val}]"""
            )

        # if one_hot:
        #     return np.concatenate(
        #         [np.array(~positive).reshape(-1, 1), np.array(positive).reshape(-1, 1)],
        #         axis=1,
        #         dtype=np.float64,
        #     )
        return self._to_numpy(positive).astype(np.float64)

    def decode(
        self,
        vals: np.ndarray[np.float64],
        denormalize: bool = True,
        return_series: bool = True,
        discretize: bool = False,
    ) -> OneDimData:
        if not np.isin(vals, [0, 1]).all():
            raise ValueError(
                f"""Incorrect value in an encoded feature {self.name}.
                All values must be either 0 or 1. Found values {np.unique(vals[~np.isin(vals, [0, 1])])}."""
            )
        vals = vals.flatten()  # TODO put the shape handlings outside, similar to encode
        res = np.empty(vals.shape, dtype=object)
        # if len(vals.shape) > 1 and vals.shape[1] > 1:
        #     res[vals[:, 0].astype(bool)] = self.__negative_val
        #     res[vals[:, 1].astype(bool)] = self.__positive_val
        # else:
        res[vals == 0] = self.__negative_val
        res[vals == 1] = self.__positive_val
        if return_series:
            return pd.Series(res, name=self.name)
        return res

    def encoding_width(self, one_hot: bool) -> int:
        # return 2 if one_hot else 1
        return 1

    def allowed_change(
        self, pre_val: CategValue, post_val: CategValue, encoded=True
    ) -> bool:
        if not encoded:
            pre_val = self.encode([pre_val], one_hot=False)[0]
            post_val = self.encode([post_val], one_hot=False)[0]
        if self.modifiable:
            if self.monotone == Monotonicity.INCREASING:
                return pre_val == self.__negative_val or post_val == self.__positive_val
            if self.monotone == Monotonicity.DECREASING:
                return pre_val == self.__positive_val or post_val == self.__negative_val
            return True
        return pre_val == post_val

    @property
    def value_mapping(self) -> dict[CategValue, int]:
        return {self.__positive_val: 1, self.__negative_val: 0}
