from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from counterfactuals.cf_methods.lice.data.Types import CategValue, OneDimData

from counterfactuals.cf_methods.lice.data.Features.Categorical import Categorical
from counterfactuals.cf_methods.lice.data.Features.Contiguous import Contiguous
from counterfactuals.cf_methods.lice.data.Features.Feature import Feature, Monotonicity


class Mixed(Feature):
    # TODO: do it via mixins?
    def __init__(
        self,
        training_vals: OneDimData,
        categ_value_names: list[CategValue],
        map_to: Optional[list[float]] = None,
        name: Optional[str] = None,
        # TODO add the bounds parameter
        default_val: float = 0,
        monotone: Monotonicity = Monotonicity.NONE,
        modifiable: bool = True,
    ):
        raise NotImplementedError("Mixed Feature is not yet tested.")
        super().__init__(training_vals, name, monotone, modifiable)
        categ_mask = np.isin(training_vals, categ_value_names)
        self.__categ_value_names = categ_value_names
        if map_to is None:
            map_to = -np.arange(len(categ_value_names)) - 1
        self.__categ_feat = Categorical(
            training_vals[categ_mask],
            categ_value_names,
            map_to,
            name,
            monotone,
            modifiable,
        )
        self.__cont_feat = Contiguous(
            training_vals[~categ_mask], name, monotone, modifiable
        )
        self.__default_val = default_val

        self._MAD = np.concatenate([self.__cont_feat.MAD, self.__categ_feat.MAD])
        # TODO, optionally make them separate into 2 columns for not-ohe
        # TODO make that into a configurable default
        # TODO, optionally give the range of applicable values (also contiguous)
        # TODO somehow makes sure that the contiguous part is >= 0

    @Feature._check_dims_on_encode
    def encode(
        self, vals: OneDimData, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        dimension = (1 + self.__categ_feat.n_categorical_vals) if one_hot else 1
        res = np.zeros(
            (vals.shape[0], dimension),
            dtype=np.float64,
        )

        categ_mask = np.isin(vals, self.__categ_value_names)
        res[~categ_mask, 0] = self.__cont_feat.encode(
            vals[~categ_mask], normalize, one_hot
        )
        if one_hot:
            res[categ_mask, 0] = self.__default_val
            res[categ_mask, 1:] = self.__categ_feat.encode(
                vals[categ_mask], normalize, one_hot
            )
        else:
            res[categ_mask, 0] = self.__categ_feat.encode(
                vals[categ_mask], normalize, one_hot
            )
        return res.astype(np.float64)

    def decode(
        self,
        vals: np.ndarray[np.float64],
        denormalize: bool = True,
        return_series: bool = True,
        discretize: bool = False,
    ) -> OneDimData:
        is_one_hot = len(vals.shape) > 1 and vals.shape[1] > 1

        res = np.empty((vals.shape[0],), dtype=object)
        if is_one_hot:
            categ_mask = vals[:, 1].astype(bool)
            for i in range(2, vals.shape[1]):
                categ_mask |= vals[:, i].astype(bool)
            res[categ_mask] = self.__categ_feat.decode(
                vals[:, 1:], denormalize, return_series=False, discretize=discretize
            )[categ_mask]
            cont_scope = vals[:, 0]
        else:
            categ_mask = np.isin(vals, list(self.__categ_feat.value_mapping.values()))
            res[categ_mask] = self.__categ_feat.decode(
                vals[categ_mask],
                denormalize,
                return_series=False,
                discretize=discretize,
            )
            cont_scope = vals
        res[~categ_mask] = self.__cont_feat.decode(
            cont_scope, denormalize, return_series=False, discretize=discretize
        )[~categ_mask]

        if return_series:
            return pd.Series(res, name=self.name)
        return res
        # TODO could be smarter, if this is an inner function, and these series wrappers are written only once... Or remove the series part altogether

    def encoding_width(self, one_hot: bool) -> int:
        if one_hot:
            return 1 + self.__categ_feat.encoding_width(one_hot)
        return 1

    @property
    def default_val(self):
        return self.__default_val

    @property
    def default_val_normalized(self):
        return self.__cont_feat.encode(
            np.array([self.__default_val]), normalize=True, one_hot=False
        )[0]

    @property
    def bounds(self):
        return self.__cont_feat.bounds

    @property
    def value_mapping(self):
        return self.__categ_feat.value_mapping

    @property
    def n_categorical_vals(self):
        return self.__categ_feat.n_categorical_vals

    @property
    def orig_vals(self):
        return self.__categ_feat.orig_vals

    @property
    def numeric_vals(self):
        return self.__categ_feat.numeric_vals
