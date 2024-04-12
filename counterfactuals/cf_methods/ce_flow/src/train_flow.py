import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactual_explanation.flow_ssl import FlowLoss
from counterfactual_explanation.flow_ssl.distributions import SSLGaussMixture
from counterfactual_explanation.flow_ssl.realnvp.coupling_layer import (
    DequantizationOriginal)
from counterfactual_explanation.flow_ssl.realnvp.realnvp import RealNVPTabular
from counterfactual_explanation.utils.data_catalog import (
    DataCatalog, EncoderNormalizeDataCatalog, LabelEncoderNormalizeDataCatalog,
    TensorDatasetTraning)
# from counterfactual_explanation.flow_ce.flow_method import (
#     CounterfactualSimpleBn, FindCounterfactualSample)
# from counterfactual_explanation.models.classifier import Net
from counterfactual_explanation.utils.helpers import (
    load_configuration_from_yaml)
from counterfactual_explanation.utils.mlcatalog import (
    load_pytorch_prediction_model_from_model_path, model_prediction,
    save_pytorch_model_to_model_path)

if __name__ == "__main__":
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

    predictive_model_path = configuration_for_proj['predictive_model_' + DATA_NAME]
    flow_model_path = configuration_for_proj['flow_model_' + DATA_NAME]

    predictive_model = load_pytorch_prediction_model_from_model_path(predictive_model_path)
    predictive_model = predictive_model.cuda()

    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target
    feature_names = encoder_normalize_data_catalog.categoricals + encoder_normalize_data_catalog.continous

    # data_frame = encoder_normalize_data_catalog.data_frame
    # target = encoder_normalize_data_catalog.target
    features = data_frame.drop(
        columns=[target], axis=1).values.astype(np.float32)
    features = torch.Tensor(features)
    features = features.cuda()

    predictions = model_prediction(predictive_model, features)

    # print(predictions)
    # negative_index = negative_prediction_index(predictions)
    # negative_instance_features = prediction_instances(
    #     features, negative_index)
    # positive_index = positive_prediction_index(predictions)
    # positive_instance_features = prediction_instances(
    #     features, positive_index)

    LR_INIT = 1e-2
    EPOCHS = 500
    BATCH_SIZE = 128
    PRINT_FREQ = 2
    MEAN_VALUE = 0.5

    if DATA_NAME == 'simple_bn':
        x1_mean = data_frame['x1'].median()
        x2_mean = data_frame['x2'].median()
        x3_mean = data_frame['x3'].median()
        means = torch.tensor([
            np.array([x1_mean, x2_mean, x3_mean]).astype(np.float32)
        ])
    elif DATA_NAME == 'moon':
        x1_mean = data_frame['x1'].median()
        x2_mean = data_frame['x2'].median()
        means = torch.tensor([
            np.array([x1_mean, x2_mean]).astype(np.float32)
        ])
    elif DATA_NAME == 'adult':
        x1_mean = 0.05
        x2_mean = 0.05
        x3_mean = 0.05
        x4_mean = data_frame['age'].mean()
        x5_mean = data_frame['hours_per_week'].mean()
        means = torch.tensor([
            np.array([x1_mean, x2_mean, x3_mean, x4_mean, x5_mean]).astype(np.float32)
        ])

        # x1_mean = data_frame['education'].median()
        # x2_mean = data_frame['marital_status'].median()
        # x3_mean = data_frame['occupation'].median()

        # means = torch.tensor([
        #     np.array([x1_mean, x2_mean, x3_mean]).astype(np.float32)
        #     ])

    prior = SSLGaussMixture(means=means, device='cuda')
    # features = data_frame[feature_names].values.astype(np.float32)
    features = data_frame[['education', 'marital_status', 'occupation', 'age', 'hours_per_week']].values.astype(
        np.float32)
    labels = data_frame[target].values.astype(np.float32)
    # labels = predictions.reshape(-1).detach().numpy()
    # labels = (predictions>0.5).float().detach().cpu()

    batch_size = 128
    features = torch.Tensor(features)
    labels = torch.Tensor(labels).reshape(-1, 1)
    train_data = torch.hstack((features, labels))
    train_data = TensorDatasetTraning(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # deq = Dequantization()
    deq = DequantizationOriginal()
    flow = RealNVPTabular(num_coupling_layers=3, in_dim=5, num_layers=5, hidden_dim=8).cuda()
    loss_fn = FlowLoss(prior)
    # optimizer = torch.optim.Adam(flow.parameters(), lr=LR_INIT, weight_decay=1e-3)
    optimizer = torch.optim.SGD(flow.parameters(), lr=LR_INIT, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # sldj = torch.zeros(batch_size).cuda()

    sldj_deq = torch.zeros(
        1,
    ).cuda()

    cur_lr = scheduler.optimizer.param_groups[0]['lr']

    print("Learning rate ", cur_lr)

    for t in range(EPOCHS):
        for local_batch, local_labels in train_loader:
            local_batch = local_batch.cuda()
            discrete_batch = local_batch[:, :3]
            continuous_batch = local_batch[:, 3:]
            local_z, sldj_deq = deq(discrete_batch, sldj_deq, reverse=False)
            local_z = torch.hstack([local_z, continuous_batch])
            local_z = flow(local_z)
            z = flow(local_batch)
            sldj = flow.logdet()
            flow_loss = loss_fn(local_z, sldj, local_labels)
            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

        cur_lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step()

        if t % PRINT_FREQ == 0:
            print('iter %s:' % t, 'loss = %.3f' % flow_loss, 'learning rate: %s' % cur_lr)

    # for local_batch, local_labels in train_loader:
    #     local_batch = local_batch.cuda()
    #     break 

    save_pytorch_model_to_model_path(flow, configuration_for_proj['flow_model_' + DATA_NAME])

    ## test
    # dequant_module = Dequantization()

    # discrete_value = local_batch[:,:3]
    # continuous_transformation, _ = deq.forward(discrete_value, ldj=None, reverse=False)
    # continuous_value = local_batch[:, 3:]
    # continuous_representation = torch.hstack([continuous_transformation, continuous_value])

    # z_value = flow(continuous_representation)
    # continuous_representation_ = flow.inverse(z_value)
    # continuous_transformation_ = continuous_representation_[:,:3]
    # continuous_value_ = continuous_representation_[:, 3:]
    # discrete_value_, _ = deq.forward(continuous_transformation_, ldj=None, reverse=True)
    # input_value = torch.hstack([discrete_value_, continuous_value_])

    # local_z = deq(local_batch)
    # local_z = flow(local_batch)
    # re_local_batch = flow.inverse(local_z)
    exit()
