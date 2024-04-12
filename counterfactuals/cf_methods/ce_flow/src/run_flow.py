import argparse
import timeit
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from carla.evaluation.benchmark import Benchmark
from counterfactual_explanation.flow_ce.flow_method import (
    CounterfactualAdult, CounterfactualSimpleBn, FindCounterfactualSample)
from counterfactual_explanation.flow_ssl.realnvp.coupling_layer import (
    Dequantization, DequantizationOriginal)
from counterfactual_explanation.models.classifier import Net
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog,
    TensorDatasetTraning)
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    find_latent_mean_two_classes, model_prediction, negative_prediction_index,
    positive_prediction_index, prediction_instances)

if __name__ == '__main__':
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='simple_bn')
    parser.add_argument('--weight', type=float, default=0.5)

    """Load parse argument"""
    args = parser.parse_args()
    # random_state = args.random_state
    weight = args.weight
    DATA_NAME = args.data_name
    # DATA_NAME = "moon"
    # DATA_NAME = "adult"

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
        deq = DequantizationOriginal()

    predictive_model, flow_model, _, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME)

    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    features = data_frame.drop(
        columns=[target], axis=1).values.astype(np.float32)
    features = torch.Tensor(features)
    features = features.cuda()

    predictions = model_prediction(predictive_model, features)
    negative_index = negative_prediction_index(predictions)
    negative_instance_features = prediction_instances(
        features, negative_index)

    positive_index = positive_prediction_index(predictions)
    positive_instance_features = prediction_instances(
        features, positive_index)

    factual_sample = negative_instance_features[0:2, :]

    mean_z0, mean_z1 = find_latent_mean_two_classes(
        flow_model, negative_instance_features, positive_instance_features)

    result_path = ''
    if DATA_NAME == 'simple_bn':
        counterfactual_instance = CounterfactualSimpleBn(
            predictive_model, flow_model, mean_z0, mean_z1, weight)
        result_path = configuration_for_proj['result_simple_bn']
    elif DATA_NAME == 'adult':
        counterfactual_instance = CounterfactualAdult(
            predictive_model, flow_model, mean_z0, mean_z1, weight, deq)
        result_path = configuration_for_proj['result_adult']

    # Run algorithm
    start = timeit.default_timer()
    cf_sample = []
    for single_factual in factual_sample:
        counterfactual = counterfactual_instance.find_counterfactual_via_optimizer(
            single_factual.reshape(1, -1)).cpu().detach().numpy()
        cf_sample.append(counterfactual)
    stop = timeit.default_timer()
    run_time = stop - start

    # Output
    factual_df = pd.DataFrame(factual_sample.cpu().detach(
    ).numpy(), columns=list(data_frame.columns)[:-1])
    counterfactual_df = pd.DataFrame(
        np.array(cf_sample), columns=list(data_frame.columns)[:-1])
    cf_sample = torch.Tensor(cf_sample).cuda()

    factual_df[target] = ''
    factual_df[target] = model_prediction(
        predictive_model, factual_sample).cpu().detach().numpy()

    counterfactual_df[target] = ''
    counterfactual_df[target] = model_prediction(
        predictive_model, cf_sample).cpu().detach().numpy()

    benchmark_instance = Benchmark(encoder_normalize_data_catalog, predictive_model,
                                   factual_df, counterfactual_df, run_time, counterfactual_df[target].values)
    benchmark_distance = benchmark_instance.compute_distances()
    benchmark_time = benchmark_instance.compute_average_time()
    benchmark_rate = benchmark_instance.compute_success_rate()
    benchmark_violation = benchmark_instance.compute_constraint_violation()
    benchmark_redundancy = benchmark_instance.compute_redundancy()
    # benchmark_nearest = benchmark_instance.compute_ynn()

    print(benchmark_violation)
    print(benchmark_redundancy)

    record_result = pd.DataFrame()

    mean_value = benchmark_distance.mean(axis=0)

    record_result['Distance_1'] = [mean_value['Distance_1']]
    record_result['Distance_2'] = [mean_value['Distance_2']]
    record_result['Distance_3'] = [mean_value['Distance_3']]
    record_result['Distance_4'] = [mean_value['Distance_4']]
    record_result['Success_Rate'] = benchmark_rate['Success_Rate'].values
    print(record_result)

    record_result.to_csv(result_path.format(
        'flow-weight-{}-dataname-{}.csv'.format(str(weight), DATA_NAME)), index=False)
