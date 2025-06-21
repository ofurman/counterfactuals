import torch
import numpy as np
from counterfactuals.cf_methods.base import BaseCounterfactual
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


class PPCEFR(BaseCounterfactual):
    def __init__(
        self,
        gen_model,
        disc_model,
        disc_model_criterion,
        device=None,
        neptune_run=None,
    ):
        # Initialize properly like PPCEF
        self.disc_model_criterion = disc_model_criterion
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.device = device if device is not None else "cpu"
        self.gen_model.to(self.device)
        self.disc_model.to(self.device)
        self.neptune_run = neptune_run

    def search_step(
        self, x_param, x_origin, contexts_origin, context_target, **search_step_kwargs
    ) -> dict:
        """Search step for the cf search process.
        :param x_param: point to be optimized
        :param x_origin: original point
        :param context_target: target context
        :param search_step_kwargs: dict with additional parameters
        :return: dict with loss and additional components to log.
        """
        alpha = search_step_kwargs.get("alpha", None)
        delta = search_step_kwargs.get("delta", None)
        categorical_features_lists = search_step_kwargs.get(  #  noqa: F841
            "categorical_features_lists", None
        )
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if delta is None:
            raise ValueError("Parameter 'delta' should be in kwargs")

        dist = torch.linalg.norm(x_origin - x_param, axis=1)

        disc_logits = self.disc_model.forward(x_param)
        disc_logits = (
            disc_logits.reshape(-1) if disc_logits.shape[0] == 1 else disc_logits
        )
        context_target = (
            context_target.reshape(-1)
            if context_target.shape[0] == 1
            else context_target
        )
        loss_disc = self.disc_model_criterion(disc_logits, context_target.float())

        p_x_param_c_target = self.gen_model(
            x_param, context=context_target.type(torch.float32)
        )
        max_inner = torch.nn.functional.relu(delta - p_x_param_c_target)

        loss = dist + alpha * (loss_disc + max_inner)
        return {
            "loss": loss,
            "dist": dist,
            "max_inner": max_inner,
            "loss_disc": loss_disc,
        }

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
        """
        Search counterfactual explanations for the given dataloader.
        """
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

        if self.disc_model:
            self.disc_model = self.disc_model.eval()
            for param in self.disc_model.parameters():
                param.requires_grad = False

        x_cfs = []
        y_cf_targets = []
        x_origs = []
        y_origs = []

        for x_origin, contexts_origin in dataloader:
            x_origin = x_origin.to(self.device)
            contexts_origin = contexts_origin.to(self.device)
            contexts_origin = contexts_origin.reshape(-1, 1)
            context_target = np.clip(contexts_origin + target_change, 0, 1)

            x_param = torch.as_tensor(x_origin).clone()
            x_param.requires_grad = True
            x_origin.requires_grad = False
            contexts_origin.requires_grad = False
            context_target.requires_grad = False
            optimizer = optim.Adam([x_param], lr=lr)
            loss_components_logging = {}

            for _ in (epoch_pbar := tqdm(range(epochs))):
                optimizer.zero_grad()
                loss_components = self.search_step(
                    x_param,
                    x_origin,
                    contexts_origin,
                    context_target,
                    **search_step_kwargs,
                )
                mean_loss = loss_components["loss"].mean()
                mean_loss.backward()
                optimizer.step()

                for loss_name, loss in loss_components.items():
                    loss_components_logging.setdefault(
                        f"cf_search/{loss_name}", []
                    ).append(loss.mean().detach().cpu().item())
                    if self.neptune_run:
                        self.neptune_run[f"cf_search/{loss_name}"].append(
                            loss.mean().detach().cpu().numpy()
                        )

                disc_loss = loss_components["loss_disc"].detach().cpu().mean().item()
                prob_loss = loss_components["max_inner"].detach().cpu().mean().item()
                epoch_pbar.set_description(
                    f"Discriminator loss: {disc_loss:.4f}, Prob loss: {prob_loss:.4f}"
                )
                if disc_loss < patience_eps and prob_loss < patience_eps:
                    break
            print(x_param[:5])
            print(x_origin[:5])
            x_cfs.append(x_param.detach().cpu().numpy())
            x_origs.append(x_origin.detach().cpu().numpy())
            y_origs.append(contexts_origin.detach().cpu().numpy())
            y_cf_targets.append(context_target.detach().cpu().numpy())

        return (
            np.concatenate(x_cfs, axis=0),
            np.concatenate(x_origs, axis=0),
            np.concatenate(y_origs, axis=0),
            np.concatenate(y_cf_targets, axis=0),
            loss_components_logging,
        )
