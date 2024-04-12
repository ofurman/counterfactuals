import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactual_explanation.models.classifier import train_predictive_model, Net
from counterfactual_explanation.utils.data_catalog import (DataCatalog, EncoderNormalizeDataCatalog,
                                                           LabelEncoderNormalizeDataCatalog,
                                                           TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from counterfactual_explanation.utils.mlcatalog import save_pytorch_model_to_model_path

if __name__ == '__main__':
    # DATA_NAME = "simple_bn"
    # DATA_NAME = "moon"
    DATA_NAME = "adult"

    CONFIG_PATH = "/home/trduong/Data/fairCE/configuration/data_catalog.yaml"
    CONFIG_FOR_PROJECT = "/home/trduong/Data/fairCE/configuration/project_configurations.yaml"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']
    MODEL_PATH = configuration_for_proj['predictive_model_' + DATA_NAME]

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    if DATA_NAME == 'simple_bn':
        encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(
            data_catalog)
    elif DATA_NAME == "adult":
        encoder_normalize_data_catalog = LabelEncoderNormalizeDataCatalog(
            data_catalog)

    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target

    labels = data_frame[target].values.astype(np.float32)
    features = data_frame.drop(
        columns=[target], axis=1).values.astype(np.float32)

    features = torch.Tensor(features)
    labels = torch.Tensor(labels).reshape(-1, 1)
    train_data = torch.hstack((features, labels))
    train_data = TensorDatasetTraning(train_data)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = Net(features.shape[1])
    model.to(DEVICE)
    model = train_predictive_model(train_loader, model)

    save_pytorch_model_to_model_path(model, MODEL_PATH)
