import os
from typing import Dict

import numpy as np
import torch
import yaml

from .data_catalog import DataCatalog, LabelEncoderNormalizeDataCatalog, EncoderNormalizeDataCatalog, load_target_features_name
from .mlcatalog import load_pytorch_prediction_model_from_model_path


def load_configuration_from_yaml(config_path):
    with open(config_path, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return conf


def load_hyperparameter_for_method(path, method, data_name) -> Dict:
    setup_catalog = load_configuration_from_yaml(path)
    hyperparameter = setup_catalog['recourse_methods'][method]["hyperparams"]
    hyperparameter["data_name"] = data_name
    return hyperparameter


# def load_target_features_name(filename, dataset, keys):
#     with open(os.path.join(filename), "r") as file_handle:
#         catalog = yaml.safe_load(file_handle)

#     for key in keys:
#         if catalog[dataset][key] is None:
#             catalog[dataset][key] = []

#     return catalog[dataset]


def load_all_configuration_with_data_name(DATA_NAME):
    CONFIG_PATH = '/home/trduong/Data/fairCE/configuration/data_catalog.yaml'
    CONFIG_FOR_PROJECT = '/home/trduong/Data/fairCE/configuration/project_configurations.yaml'
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']

    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)

    if DATA_NAME == 'simple_bn':
        encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(data_catalog)
    elif DATA_NAME == "adult":
        encoder_normalize_data_catalog = LabelEncoderNormalizeDataCatalog(
            data_catalog)


    predictive_model_path = configuration_for_proj['predictive_model_' + DATA_NAME]
    flow_model_path = configuration_for_proj['flow_model_' + DATA_NAME]
    
    predictive_model = load_pytorch_prediction_model_from_model_path(predictive_model_path)
    predictive_model = predictive_model.cuda()

    flow_model = load_pytorch_prediction_model_from_model_path(flow_model_path)
    flow_model = flow_model.cuda()

    return predictive_model, flow_model, encoder_normalize_data_catalog, configuration_for_proj
