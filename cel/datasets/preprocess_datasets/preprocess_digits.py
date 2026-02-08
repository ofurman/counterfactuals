import pandas as pd
from sklearn.datasets import load_digits


def preprocess_digits() -> pd.DataFrame:
    data = load_digits(n_class=10)
    pd_data = pd.DataFrame(data.data, columns=data.feature_names)
    pd_data["target"] = data.target
    return pd_data


if __name__ == "__main__":
    pd_data = preprocess_digits()
    pd_data.to_csv("data/digits.csv", index=False)
