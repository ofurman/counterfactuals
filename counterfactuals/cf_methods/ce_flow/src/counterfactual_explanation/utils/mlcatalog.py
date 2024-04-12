import torch
from .data_catalog import DataCatalog

def load_pytorch_prediction_model_from_model_path(model_path):
    # print("Load model")
    model = torch.load(model_path)
    model.eval()
    # print("End load")
    return model


def load_pytorch_prediction_model_with_state_dict(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_pytorch_model_to_model_path(model, model_path):
    torch.save(model, model_path)


def get_latent_representation_from_flow(flow, input_value):
    return flow(input_value)


def original_space_value_from_latent_representation(flow, z_value):
    return flow.inverse(z_value)

def get_latent_representation_from_flow_mixed_type(flow, deq, input_value, index_):
    discrete_value = input_value[:,:index_]
    continuous_transformation, _ = deq.forward(discrete_value, ldj=None, reverse=False)
    continuous_value = input_value[:, index_:]
    continuous_representation = torch.hstack([continuous_transformation, continuous_value])

    # z_value = flow(continuous_representation)
    # continuous_value_ = flow.inverse(z_value)
    # discrete_value_ = z_value[:,:index_]
    # continuous_value_ = z_value[:, index_:]
    # discrete_value_, _ = deq.forward(discrete_value_, ldj=None, reverse=True)
    # input_value = torch.hstack([discrete_value_, continuous_value_])


    return flow(continuous_representation)

def original_space_value_from_latent_representation_mixed_type(flow, deq, z_value, index_):
    continuous_value = flow.inverse(z_value)
    discrete_value = continuous_value[:,:index_]
    continuous_value = continuous_value[:, index_:]
    discrete_value, _ = deq.forward(discrete_value, ldj=None, reverse=True)
    input_value = torch.hstack([discrete_value, continuous_value])

    return input_value

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



def model_prediction(predictive_model, features):
    return predictive_model(features)

def negative_prediction_index(prediction):
    return torch.lt(prediction, 0.5).reshape(-1)

def positive_prediction_index(prediction):
    return torch.gt(prediction, 0.5).reshape(-1)

def prediction_instances(instances, indexes):
    return instances[indexes]


def find_latent_mean_two_classes(flow, x0, x1):
    z0 = flow(x0) 
    z1 = flow(x1)
    mean_z0 = torch.mean(z0)
    mean_z1 = torch.mean(z1)
    return mean_z0, mean_z1

class MLModelCatalog():
    def __init__(self,data: DataCatalog, predictive_model) -> None:
        self.model = predictive_model 
        self._continuous = data.continous
        self._categoricals = data.categoricals

    def predict(self):
        pass
    def predict_proba(self):
        pass


def train_one_epoch_batch_data(flow_model, optimizer, loss_fn, batch_x, batch_y):
    z_value = flow_model(batch_x)
    sldj = flow_model.logdet()
    loss = loss_fn(z_value, sldj, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss