import numpy as np
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.datasets.file_dataset import FileDataset
from counterfactuals.models import (
    KDE,
    NICE,
    MaskedAutoregressiveFlow,
    RealNVP,
)


def test_kde():
    model = KDE(bandwidth=0.1)
    assert model.bandwidth == 0.1


def test_masked_autoregressive_flow():
    model = MaskedAutoregressiveFlow(features=10, hidden_features=10)
    assert model.features == 10
    assert model.hidden_features == 10


def test_nice():
    model = NICE(features=10, hidden_features=10)
    assert model.features == 10
    assert model.hidden_features == 10


def test_real_nvp():
    model = RealNVP(features=10, hidden_features=10)
    assert model.features == 10
    assert model.hidden_features == 10


def prepare_moons_dataset():
    dataset = FileDataset(config_path="config/datasets/moons.yaml")
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
    return dataset


def test_kde_fit():
    dataset = prepare_moons_dataset()
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
    model = KDE(features=dataset.X_train.shape[1], hidden_features=10)
    model.fit(train_dataloader, test_dataloader, epochs=10, lr=0.001)
    predictions = model.predict_log_proba(dataset.X_test, context=dataset.y_test)
    assert predictions.shape == (dataset.X_test.shape[0],)
    model.save("test_model.pth")
    model.load("test_model.pth")


@pytest.mark.parametrize("model_class", [MaskedAutoregressiveFlow, NICE, RealNVP])
def test_model_fit_binary_without_context(
    model_class: type[MaskedAutoregressiveFlow | NICE | RealNVP],
):
    dataset = prepare_moons_dataset()
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
    # model without context
    model = model_class(features=dataset.X_train.shape[1], hidden_features=10)
    model.fit(train_dataloader, test_dataloader, epochs=10, lr=0.001)

    predictions = model.predict_log_proba(dataset.X_test)
    predictions_sample, predictions_log_proba = model.sample_and_log_proba(n_samples=10)
    assert predictions.shape == (dataset.X_test.shape[0],)
    assert predictions_sample.shape == (10, dataset.X_test.shape[1])
    assert predictions_log_proba.shape == (10,)
    model.save("test_model.pth")
    model.load("test_model.pth")


# disable for now because of context support
@pytest.mark.parametrize("model_class", [MaskedAutoregressiveFlow, NICE, RealNVP])
def test_model_fit_binary_with_context(
    model_class: type[MaskedAutoregressiveFlow | NICE | RealNVP],
):
    dataset = prepare_moons_dataset()
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

    model = model_class(
        features=dataset.X_train.shape[1], hidden_features=10, context_features=1
    )
    model.fit(train_dataloader, test_dataloader, epochs=10, lr=0.001)

    predictions = model.predict_log_proba(dataset.X_test, context=dataset.y_test)
    predictions_sample, predictions_log_proba = model.sample_and_log_proba(
        n_samples=10, context=np.unique(dataset.y_test)
    )
    assert predictions.shape == (dataset.X_test.shape[0],)
    assert predictions_sample.shape == (
        len(np.unique(dataset.y_test)),
        10,
        dataset.X_test.shape[1],
    )
    assert predictions_log_proba.shape == (len(np.unique(dataset.y_test)), 10)
    model.save("test_model.pth")
    model.load("test_model.pth")
