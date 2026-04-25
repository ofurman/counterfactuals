import numpy as np
import pandas as pd

from cel.cf_methods.global_methods.globe_ce import GLOBE_CE


class _Dataset:
    features = ["x0", "x1"]
    features_tree = {"x0": [], "x1": []}
    categorical_columns = []
    numerical_columns = [0, 1]


def test_globe_ce_evaluate_uses_configured_target_class_zero():
    x = pd.DataFrame([[1.0, 0.0], [2.0, 0.0]], columns=_Dataset.features)

    def predict_fn(values):
        values = values.values if isinstance(values, pd.DataFrame) else values
        return (values[:, 0] > 0.5).astype(int)

    method = GLOBE_CE(
        predict_fn=predict_fn,
        dataset=_Dataset(),
        X=x,
        bin_widths={"x0": 1.0, "x1": 1.0},
        target_class=0,
    )

    correct, cost = method.evaluate(np.array([-1.0, 0.0]))

    assert correct.tolist() == [100, 0]
    assert cost.tolist() == [1.0, 0.0]
