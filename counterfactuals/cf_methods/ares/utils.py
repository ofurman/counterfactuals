import pandas as pd
from sklearn.preprocessing import LabelEncoder

from counterfactuals.datasets.moons import MoonsDataset


def ares_one_hot(dataset):
    """
    Improvised method for one-hot encoding the data

    Input: data (whole dataset)
    """

    label_encoder = LabelEncoder()
    data_encode = pd.DataFrame(dataset.X, columns=dataset.feature_columns)
    dataset.bins = {}
    dataset.bins_tree = {}
    dataset.features_tree = {}

    # Assign encoded features to one hot columns
    features = []
    for x in data_encode.columns:
        dataset.features_tree[x] = []
        categorical = x in dataset.categorical_features
        if categorical:
            data_encode[x] = label_encoder.fit_transform(data_encode[x])
            cols = label_encoder.classes_
        elif dataset.n_bins is not None:
            data_encode[x] = pd.cut(
                data_encode[x].apply(lambda x: float(x)), bins=dataset.n_bins
            )
            cols = data_encode[x].cat.categories
            dataset.bins_tree[x] = {}
        else:
            features.append(x)
            continue

        for col in cols:
            feature_value = x + " = " + str(col)
            features.append(feature_value)
            dataset.features_tree[x].append(feature_value)
            if not categorical:
                dataset.bins[feature_value] = col.mid
                dataset.bins_tree[x][feature_value] = col.mid

    dataset.features = features


def add_method_variables(dataset, n_bins=None):
    dataset.n_bins = n_bins
    dataset.categorical_features = []

    if isinstance(dataset, MoonsDataset):
        dataset.raw_data.columns = ["0", "1", "2"]
        dataset.feature_columns = ["0", "1"]
