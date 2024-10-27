import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from counterfactuals.cf_methods.base import BaseCounterfactual
from counterfactuals.discriminative_models.base import BaseDiscModel
from counterfactuals.generative_models.base import BaseGenModel


class WACH_OURS(BaseCounterfactual):
    def __init__(
        self,
        gen_model: BaseGenModel,
        disc_model: BaseDiscModel,
        disc_model_criterion,
        device=None,
        neptune_run=None,
    ):
        self.disc_model_criterion = disc_model_criterion
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.device = device if device is not None else "cpu"
        self.gen_model.to(self.device)
        self.disc_model.to(self.device)
        self.neptune_run = neptune_run

    def _search_step(
        self, delta, x_origin, contexts_origin, context_target, **search_step_kwargs
    ) -> dict:
        """Search step for the cf search process.
        :param x_param: point to be optimized
        :param x_origin: original point
        :param context_target: target context
        :param search_step_kwargs: dict with additional parameters
        :return: dict with loss and additional components to log.
        """
        alpha = search_step_kwargs.get("alpha", None)
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")

        dist = torch.linalg.vector_norm(delta, dim=1, ord=2)

        disc_logits = self.disc_model.forward(x_origin + delta)
        disc_logits = (
            disc_logits.reshape(-1) if disc_logits.shape[0] == 1 else disc_logits
        )
        context_target = (
            context_target.reshape(-1)
            if context_target.shape[0] == 1
            else context_target
        )
        loss_disc = self.disc_model_criterion(disc_logits, context_target.float())

        loss = dist + alpha * (loss_disc)
        return {
            "loss": loss,
            "dist": dist,
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
        **search_step_kwargs,
    ):
        """
        Search counterfactual explanations for the given dataloader.
        """
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

        if self.disc_model:
            self.disc_model.eval()
            for param in self.disc_model.parameters():
                param.requires_grad = False

        deltas = []
        target_class = []
        original = []
        original_class = []
        for xs_origin, contexts_origin in dataloader:
            xs_origin = xs_origin.to(self.device)
            contexts_origin = contexts_origin.to(self.device)

            contexts_origin = contexts_origin.reshape(-1, 1)
            contexts_target = torch.abs(1 - contexts_origin)

            xs_origin = torch.as_tensor(xs_origin)
            xs_origin.requires_grad = False
            delta = torch.zeros_like(xs_origin, requires_grad=True)

            optimizer = optim.Adam([delta], lr=lr)
            loss_components_logging = {}

            for _ in (epoch_pbar := tqdm(range(epochs))):
                optimizer.zero_grad()
                loss_components = self._search_step(
                    delta,
                    xs_origin,
                    contexts_origin,
                    contexts_target,
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
                epoch_pbar.set_description(f"Discriminator loss: {disc_loss:.4f}")
                if disc_loss < patience_eps:
                    break

            deltas.append(delta.detach().cpu().numpy())
            original.append(xs_origin.detach().cpu().numpy())
            original_class.append(contexts_origin.detach().cpu().numpy())
            target_class.append(contexts_target.detach().cpu().numpy())
        return (
            np.concatenate(deltas, axis=0),
            np.concatenate(original, axis=0),
            np.concatenate(original_class, axis=0),
            np.concatenate(target_class, axis=0),
            loss_components_logging,
        )
