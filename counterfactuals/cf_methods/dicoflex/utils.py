from typing import List

import numpy as np
import torch


def transform_batch_data(data: List, device: str):
    data = torch.stack(data)
    data = data.reshape(-1, data.shape[-1])
    return data.to(device).float()


def inverse_transform_data(data, dataset):
    data[:, dataset.categorical_features] = dataset.qt.inverse_transform(data[:, dataset.categorical_features])
    data_orig = np.empty((
        len(data), len(dataset.numerical_columns) + len(dataset.categorical_columns)
    ), dtype=object)

    numerical_pos = len(dataset.numerical_columns)
    numerical_indexes = [
        dataset.train_data.columns.get_loc(feat) for feat in dataset.feature_columns[:numerical_pos]
    ]
    data_orig[:, numerical_indexes] = (
        dataset.feature_transformer.named_transformers_["MinMaxScaler"].inverse_transform(
            data[:, dataset.numerical_features])
    )

    categorical_indexes = [
        dataset.train_data.columns.get_loc(feat) for feat in dataset.feature_columns[numerical_pos:]
    ]
    data_orig[:, categorical_indexes] = (
        dataset.feature_transformer.named_transformers_["OneHotEncoder"].inverse_transform(
            data[:, dataset.categorical_features]
        )
    )

    return data_orig
