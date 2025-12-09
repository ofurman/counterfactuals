import numpy as np
import torch
import torch.optim as optim
import GPyOpt
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm

from counterfactuals.cf_methods.counterfactual_base import BaseCounterfactualMethod
from counterfactuals.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)


class CEARM(BaseCounterfactualMethod, LocalCounterfactualMixin):
    def __init__(
        self,
        disc_model,
        device=None,
    ):
        self.disc_model = disc_model
        self.device = device if device is not None else "cpu"
        self.disc_model.to(self.device)

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """
        Explains the model's prediction for a given input.
        """
        raise NotImplementedError("This method is not implemented for this class.")

    def explain_dataloader(
        self,
        dataloader: DataLoader,
        epochs: int = 1000,
        lr: float = 0.0005,
        patience_eps: int = 1e-5,
        target_change: float = 0.2,
        **search_step_kwargs,
    ):
        X_test, y_test = dataloader.dataset.tensors

        # Define the EP potential function
        def ep_potential(y, y_query, w=0.2):
            term = (y_query - y) / w
            z = np.maximum(term, 0)
            aep_p = z**2 * np.exp(-(z**2))
            # z = - np.minimum(term, 0)
            # aep_m = z ** 2 * np.exp(-(z**2))
            return aep_p
            # return (y - y_query)**2 * np.exp(-((y - y_query)**2) / w)

        # Objective function for Bayesian Optimization
        def objective_function(X_new, index):
            X_new = X_new.reshape(1, -1)
            y_query = self.disc_model.predict(
                torch.from_numpy(X_test[index].reshape(1, -1))
            ).numpy()  # Current prediction we want to change
            y_pred = self.disc_model.predict(X_new).numpy()
            potential = ep_potential(y_query, y_pred)
            return potential

        start_time = time()

        Xs_cfs = []
        model_returned = []
        y_cf_targets = []
        x_origs = []
        y_origs = []

        for i in tqdm(range(len(X_test))):
            domain = [
                {"name": f"var_{j}", "type": "continuous", "domain": (0, 1)}
                for j in range(X_test.shape[1])
            ]
            optimizer = GPyOpt.methods.BayesianOptimization(
                f=lambda X_new: objective_function(X_new, i),
                domain=domain,
                model_type="GP",
                acquisition_type="EI",
                normalize_Y=False,
                evaluator_type="thompson_sampling",
                maximize=True,
            )
            optimizer.run_optimization(eps=1e-3, max_iter=5)
            Xs_cfs.append(optimizer.x_opt.reshape(1, -1))
            model_returned.append(True)

        cf_search_time = time() - start_time
        Xs_cfs = np.array(Xs_cfs, dtype=np.float32).squeeze()
        x_origs = X_test.detach().cpu().numpy()
        y_origs = y_test.detach().cpu().numpy()
        y_cf_targets = y_test.detach().cpu().numpy()

        return (
            np.concatenate(Xs_cfs, axis=0),
            np.concatenate(x_origs, axis=0),
            np.concatenate(y_origs, axis=0),
            np.concatenate(y_cf_targets, axis=0),
            None,
        )
