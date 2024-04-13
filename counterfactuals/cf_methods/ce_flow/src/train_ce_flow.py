import numpy as np
import torch

from counterfactual_explanation.flow_ssl.distributions import SSLGaussMixture
from counterfactual_explanation.utils.data_catalog import (DataCatalog, EncoderNormalizeDataCatalog,
                                                           TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import load_configuration_from_yaml
from torch.utils.data import DataLoader
from counterfactual_explanation.utils.helpers import \
    load_all_configuration_with_data_name
from counterfactual_explanation.utils.mlcatalog import (
    model_prediction, negative_prediction_index, positive_prediction_index, prediction_instances)



if __name__ == "__main__":
    DATA_NAME = 'simple_bn'
    CONFIG_PATH = '/home/trduong/Data/fairCE/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/home/trduong/Data/fairCE/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(data_catalog)
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.categoricals + \
        encoder_normalize_data_catalog.continous
    predictive_model, _, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME)

    LR_INIT = 1e-6
    EPOCHS = 200
    BATCH_SIZE = 32
    PRINT_FREQ = 10
    MEAN_VALUE = 0.5

    if DATA_NAME == 'simple_bn':
        x1_mean = data_frame['x1'].median()
        x2_mean = data_frame['x2'].median()
        x3_mean = data_frame['x3'].median()
        means = torch.tensor([
            np.array([x1_mean, x2_mean, x3_mean]).astype(np.float32)
        ])

    prior = SSLGaussMixture(means=means, device='cuda')

    features = data_frame[feature_names].values.astype(np.float32)
    features = torch.Tensor(features)
    features_cuda = features
    labels = model_prediction(predictive_model, features_cuda).detach().cpu()

    negative_index = negative_prediction_index(labels)
    negative_instance_features = prediction_instances(features, negative_index)
    negative_labels = prediction_instances(labels, negative_index)
    negative_data = torch.hstack(
        (negative_instance_features, negative_labels))
    negative_data = TensorDatasetTraning(negative_data)
    negative_loader = DataLoader(negative_data, batch_size=64, shuffle=True)

    positive_index = positive_prediction_index(labels)
    positive_instance_features = prediction_instances(features, positive_index)
    positive_labels = prediction_instances(labels, positive_index)
    positive_data = torch.hstack(
        (positive_instance_features, positive_labels))
    positive_data = TensorDatasetTraning(positive_data)
    positive_loader = DataLoader(positive_data, batch_size=64, shuffle=True)
