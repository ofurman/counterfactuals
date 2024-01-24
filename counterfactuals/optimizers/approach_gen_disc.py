"""
# Approach 2

$$agmin\quad d(x, x') - \lambda (log p(x'|y') - log(p(x'|y) + p(x'|y')))$$
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial
from sklearn.base import RegressorMixin, ClassifierMixin

from counterfactuals.optimizers.base import BaseCounterfactualModel


class ApproachGenDisc(BaseCounterfactualModel):
    def search_step(self, x_param, x_origin, context_origin, context_target, **kwargs):
        """
        Performs a single search step for counterfactual point.
        """
        alpha = kwargs.get("alpha", None)
        beta = kwargs.get("beta", None)
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if beta is None:
            raise ValueError("Parameter 'beta' should be in kwargs")

        dist = torch.linalg.norm(x_origin-x_param, axis=1)

        # assert np.all(np.argmax(self.classifier.predict_proba(x_origin), axis=1) == context_origin.numpy().astype(int))
        # p_y_x_param = torch.from_numpy(np.max(self.classifier.predict_proba(x_param.detach()), axis=1))
        # p_y_x_orig = torch.from_numpy(np.max(self.classifier.predict_proba(x_origin), axis=1))

        p_pred = self.classifier.predict_proba(x_origin)
        max_i = np.argmax(p_pred, axis=1)
        max_i = np.vstack([np.arange(p_pred.shape[0]), max_i])

        p_y_x_param = torch.from_numpy(self.classifier.predict_proba(x_param.detach())[*max_i])
        p_y_x_orig = torch.from_numpy(self.classifier.predict_proba(x_origin)[*max_i])

        p_x_param = self.model.log_prob(x_param, context=None).exp()
        p_x_orig = self.model.log_prob(x_param, context=None).exp()

        # p(x'|y) = p(y|x')p(x')
        p_x_param_c_orig = p_y_x_param * p_x_param
        # p(x'|y') = p(y'|x')p(x')
        p_x_param_c_target = (1 - p_y_x_param) * p_x_param
        # p(x|y) = p(y|x)p(x)
        p_x_orig_c_orig = p_y_x_orig * p_x_orig

        p_x_param_c_orig_with_beta = p_x_param_c_orig + beta
        max_inner = torch.nn.functional.relu(p_x_orig_c_orig-p_x_param_c_target)
        max_outer = torch.nn.functional.relu(p_x_param_c_orig_with_beta - p_x_param_c_target)
        loss = dist + alpha * (max_outer + max_inner)
        return loss, dist, max_inner, max_outer
    
    def generate_counterfactuals(self, Xs, ys, epochs, lr, alpha, beta):
        Xs = Xs[:, np.newaxis, :]
        ys = ys.reshape(-1, 1)
        ys_hat = np.abs(1-ys).reshape(-1, 1)
        x_cfs = []
        for X, y, y_hat in tqdm(zip(Xs, ys, ys_hat)):
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            y_hat = torch.Tensor(y_hat)
            x_cf = self.search(X, y, y_hat, num_epochs=epochs, lr=lr, alpha=alpha, beta=beta, verbose=False)
            x_cfs.append(x_cf)

        # x_cfs = np.array([x.detach().numpy() for x in x_cfs]).squeeze()
        return x_cfs