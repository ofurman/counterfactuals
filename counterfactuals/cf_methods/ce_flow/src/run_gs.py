import argparse
import timeit

import numpy as np
import pandas as pd
import torch

from carla import MLModelCatalog
from carla.evaluation.benchmark import Benchmark
# from carla.recourse_methods import GrowingSpheres
from carla.recourse_methods import *
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog)
from counterfactual_explanation.utils.helpers import (
    load_all_configuration_with_data_name, load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    model_prediction,
    negative_prediction_index, prediction_instances)

if __name__ == "__main__":
    # DATA_NAME = 'simple_bn'
    # DATA_NAME = 'adult'
    # predictive_model, flow_model, encoder_normalize_data_catalog, configuration_for_proj = load_all_configuration_with_data_name(
    #     DATA_NAME)
    # DATA_NAME = "simple_bn"
    # DATA_NAME = "moon"
    # DATA_NAME = "adult"
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='simple_bn')
    # parser.add_argument('--data_name', type=str, default='adult')

    """Load parse argument"""
    args = parser.parse_args()
    # random_state = args.random_state
    DATA_NAME = args.data_name

    CONFIG_PATH = "/home/trduong/Data/fairCE/configuration/data_catalog.yaml"
    CONFIG_FOR_PROJECT = "/home/trduong/Data/fairCE/configuration/project_configurations.yaml"
    CONFIG_FOR_EXPERIMENT = "/home/trduong/Data/fairCE/configuration/experimental_setup.yaml"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration_for_proj = load_configuration_from_yaml(CONFIG_FOR_PROJECT)
    configuration_for_expr = load_configuration_from_yaml(CONFIG_FOR_EXPERIMENT)['recourse_methods']['revise']['hyperparams']

    DATA_PATH = configuration_for_proj[DATA_NAME + '_dataset']
    MODEL_PATH = configuration_for_proj['predictive_model_' + DATA_NAME]
    


    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    if DATA_NAME == 'simple_bn':
        processed_catalog = EncoderNormalizeDataCatalog(
            data_catalog)
    elif DATA_NAME == "adult":
        processed_catalog = LabelEncoderNormalizeDataCatalog(
            data_catalog, gs=True)

    predictive_model, flow_model, _, configuration_for_proj = load_all_configuration_with_data_name(
        DATA_NAME)


    data_frame = processed_catalog.data_frame
    target = processed_catalog.target

    features = data_frame.drop(
        columns=[target], axis=1).values.astype(np.float32)
    features = torch.Tensor(features)
    features = features.cuda()

    predictions = model_prediction(predictive_model, features)
    negative_index = negative_prediction_index(predictions)
    negative_instance_features = prediction_instances(
        features, negative_index)

    position = negative_index.nonzero(
        as_tuple=False).cpu().detach().numpy().reshape(-1)

    factual_sample = data_frame.loc[position]
    factual_sample = factual_sample[:5]

    # processed_catalog.data_catalog.raw = data_frame
    model = MLModelCatalog(
        processed_catalog.data_catalog, predictive_model)
    model.raw_model.cuda()

    configuration_for_expr["data_name"] = DATA_NAME
    configuration_for_expr["vae_params"]["layers"] = [
                                                         len(model.feature_input_order)
                                                     ] + configuration_for_expr["vae_params"]["layers"]
    gs = CCHVAE(DATA_NAME, model, processed_catalog, configuration_for_expr)
    # gs = GrowingSpheres(DATA_NAME, model, processed_catalog)

    result_path = ''
    if DATA_NAME == 'simple_bn':
        result_path = configuration_for_proj['result_simple_bn']
    elif DATA_NAME == 'adult':
        result_path = configuration_for_proj['result_adult']

    start = timeit.default_timer()
    counterfactuals_gs = gs.get_counterfactuals(factual_sample)
    stop = timeit.default_timer()
    run_time = stop - start

    cf_sample = torch.Tensor(counterfactuals_gs.values).cuda()
    counterfactuals_gs[target] = model_prediction(
        predictive_model, cf_sample).cpu().detach().numpy()

    benchmark_instance = Benchmark(processed_catalog, predictive_model,
                                   factual_sample, counterfactuals_gs, run_time, counterfactuals_gs[target].values)
    benchmark_distance = benchmark_instance.compute_distances()
    benchmark_time = benchmark_instance.compute_average_time()
    benchmark_rate = benchmark_instance.compute_success_rate()

    record_result = pd.DataFrame()
    mean_value = benchmark_distance.mean(axis=0)

    record_result['Distance_1'] = [mean_value['Distance_1']]
    record_result['Distance_2'] = [mean_value['Distance_2']]
    record_result['Distance_3'] = [mean_value['Distance_3']]
    record_result['Distance_4'] = [mean_value['Distance_4']]
    record_result['Success_Rate'] = benchmark_rate['Success_Rate'].values
    # print(record_result)
    # record_result.to_csv(result_path.format('gs.csv'), index=False)
    record_result.to_csv(result_path.format('cchvae.csv'), index=False)

    print(record_result)
