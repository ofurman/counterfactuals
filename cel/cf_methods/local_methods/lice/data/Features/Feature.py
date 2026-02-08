from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from cel.cf_methods.local.lice.data.Types import OneDimData


class Monotonicity(Enum):
    INCREASING = 1
    NONE = 0
    DECREASING = -1


class Feature(ABC):
    def __init__(
        self,
        training_vals: OneDimData,
        name: Optional[str],
        monotone: Monotonicity = Monotonicity.NONE,
        modifiable: bool = True,
    ):
        if name is None:
            if isinstance(training_vals, pd.Series):
                name = str(training_vals.name)
            else:
                raise ValueError("Name of the feature must be specified in pd.Series or directly")
        if training_vals.shape[0] == 0:
            raise ValueError(f"No data provided to feature {name}")
        self.__name = name
        self.__monotone = monotone
        self.__modifiable = modifiable

    @property
    def monotone(self):
        return self.__monotone

    @property
    def modifiable(self):
        return self.__modifiable

    def _to_numpy(self, vals: OneDimData) -> np.ndarray:
        if isinstance(vals, pd.Series):
            return vals.to_numpy()
        return vals

    def _check_dims_on_encode(func):
        def dim_check(self, vals: OneDimData, *args, **kwargs):
            if isinstance(vals, np.ndarray) or isinstance(vals, pd.Series):
                # can be squeezed to 0 dims if a single value is passed
                if len(vals.shape) > 1:
                    if len(np.squeeze(vals).shape) > 1:
                        raise ValueError("Incorect dimension of feature")
                    return func(self, vals.flatten(), *args, **kwargs)
                    # TODO reintroduce the dimensions
                return func(self, vals, *args, **kwargs)
            if isinstance(vals, list):
                return list(func(self, np.array(vals), *args, **kwargs))
            # we assume it is a single value
            return func(self, np.array([vals]), *args, **kwargs)[0]

        return dim_check

    @property
    def name(self) -> str:
        return self.__name

    @property
    def MAD(self) -> np.ndarray[np.float64]:
        return self._MAD

    @abstractmethod
    def encode(
        self, vals: OneDimData, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        """Encodes the vals"""

    @abstractmethod
    def decode(
        self,
        vals: np.ndarray[np.float64],
        denormalize: bool = True,
        # one_hot: bool = True, #TODO add this too
        return_series: bool = True,
        discretize: bool = False,
    ) -> OneDimData:
        """Decodes the vals into the original form"""

    @abstractmethod
    def encoding_width(self, one_hot: bool) -> int:
        """Returns the width of the encoded values, i.e., the size in teh second dimension (axis 1)"""

    @abstractmethod
    def allowed_change(self, pre_val, post_val, encoded: bool) -> bool:
        """Checks whether value change from pre_val to post_val is allowed by mutability and similar properties"""

    # TODO do some __repr__()?
    def __str__(self):
        return str(self.__name)
