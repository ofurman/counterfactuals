from __future__ import annotations

import numpy as np
import pandas as pd

from counterfactuals.cf_methods.local.lice.data.Features.Feature import Feature, Monotonicity
from counterfactuals.cf_methods.local.lice.data.Types import OneDimData


class Contiguous(Feature):
    def __init__(
        self,
        training_vals: OneDimData,
        name: str | None = None,
        bounds: tuple[float, float] | None = None,
        discrete: bool = False,
        monotone: Monotonicity = Monotonicity.NONE,
        modifiable: bool = True,
    ):
        super().__init__(training_vals, name, monotone, modifiable)
        if isinstance(training_vals, pd.Series):
            training_vals = training_vals.to_numpy()

        if bounds is not None:
            self.__bounds = bounds
        else:
            self.__bounds = (training_vals.min(), training_vals.max())
            if self.__bounds[0] == self.__bounds[1]:
                raise ValueError(
                    f"Values of feature {self.name} have a single value and no bounds were preset"
                )
        self._shift = self.__bounds[0]
        self._scale = self.__bounds[1] - self.__bounds[0]

        self.__discrete = discrete

        normalized = self.__normalize(training_vals)
        median = np.nanmedian(normalized)
        self._MAD = np.asarray([np.nanmedian(np.abs(normalized - median))])
        if self._MAD[0] == 0:
            self._MAD[0] = 1.48 * np.nanstd(normalized)
            if self._MAD[0] == 0:
                self._MAD[0] = 1

    def __normalize(self, vals):
        return (vals - self._shift) / self._scale

    def __denormalize(self, vals):
        return vals * self._scale + self._shift

    @Feature._check_dims_on_encode
    def encode(
        self, vals: OneDimData, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        if isinstance(vals, pd.Series):
            vals = vals.to_numpy()
        if normalize:
            return self.__normalize(vals)
        return vals.astype(np.float64)

    def decode(
        self,
        vals: np.ndarray[np.float64],
        denormalize: bool = True,
        return_series: bool = True,
        discretize: bool = True,
    ) -> OneDimData:
        if denormalize:
            vals = self.__denormalize(vals)
        if discretize and self.discrete:
            vals = np.round(vals)
        if return_series:
            return pd.Series(vals.flatten(), name=self.name)
        return vals

    def encoding_width(self, one_hot: bool) -> int:
        return 1

    @property
    def bounds(self) -> tuple[float, float]:
        return self.__bounds

    @property
    def discrete(self) -> bool:
        return self.__discrete

    def allowed_change(self, pre_val: float, post_val: float, encoded=True) -> bool:
        if self.modifiable:
            if self.monotone == Monotonicity.INCREASING:
                return pre_val <= post_val
            if self.monotone == Monotonicity.DECREASING:
                return pre_val >= post_val
            return True
        return pre_val == post_val
