import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.datasets.file_dataset import FileDataset
from counterfactuals.models import (
    NODE,
    LogisticRegression,
    MLPClassifier,
    MultinomialLogisticRegression,
)


def test_mlp_classifier():
    model = MLPClassifier(num_inputs=10, num_targets=2)
    assert model.num_inputs == 10
    assert model.num_targets == 2


def test_logistic_regression():
    model = LogisticRegression(num_inputs=10, num_targets=2)
    assert model.num_inputs == 10
    assert model.num_targets == 2


def test_node():
    model = NODE(num_inputs=10, num_targets=2)
    assert model.num_inputs == 10
    assert model.num_targets == 2


@pytest.mark.parametrize("model_class", [MLPClassifier, LogisticRegression, NODE])
def test_model_fit_binary(model_class: type[MLPClassifier | LogisticRegression | NODE]):
    dataset = FileDataset(config_path="config/datasets/adult.yaml")
    feature_transformer = ColumnTransformer(
        [
            ("MinMaxScaler", MinMaxScaler(), dataset.numerical_features_indices),
            (
                "OneHotEncoder",
                OneHotEncoder(sparse_output=False),
                dataset.categorical_features_indices,
            ),
        ]
    )
    dataset.X_train = feature_transformer.fit_transform(dataset.X_train)
    dataset.X_test = feature_transformer.transform(dataset.X_test)
    dataset.y_train = dataset.y_train.astype(int)
    dataset.y_test = dataset.y_test.astype(int)
    train_dataset = TensorDataset(
        torch.tensor(dataset.X_train, dtype=torch.float32),
        torch.tensor(dataset.y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(dataset.X_test, dtype=torch.float32),
        torch.tensor(dataset.y_test, dtype=torch.float32),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    model = model_class(num_inputs=dataset.X_train.shape[1], num_targets=1)
    model.fit(train_dataloader, test_dataloader, epochs=10, lr=0.001)
    assert model.num_inputs == dataset.X_train.shape[1]


@pytest.mark.parametrize(
    "model_class", [MLPClassifier, MultinomialLogisticRegression, NODE]
)
def test_model_fit_multiclass(
    model_class: type[MLPClassifier | MultinomialLogisticRegression | NODE],
):
    dataset = FileDataset(config_path="config/datasets/blobs.yaml")
    feature_transformer = ColumnTransformer(
        [
            ("MinMaxScaler", MinMaxScaler(), dataset.numerical_features_indices),
            (
                "OneHotEncoder",
                OneHotEncoder(sparse_output=False),
                dataset.categorical_features_indices,
            ),
        ]
    )
    dataset.X_train = feature_transformer.fit_transform(dataset.X_train)
    dataset.X_test = feature_transformer.transform(dataset.X_test)
    dataset.y_train = dataset.y_train.astype(int)
    dataset.y_test = dataset.y_test.astype(int)
    train_dataset = TensorDataset(
        torch.tensor(dataset.X_train, dtype=torch.float32),
        torch.tensor(dataset.y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(dataset.X_test, dtype=torch.float32),
        torch.tensor(dataset.y_test, dtype=torch.float32),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    model = model_class(num_inputs=dataset.X_train.shape[1], num_targets=3)
    model.fit(train_dataloader, test_dataloader, epochs=10, lr=0.001)
    assert model.num_inputs == dataset.X_train.shape[1]
