import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from counterfactuals.cf_methods.base import BaseCounterfactual
from counterfactuals.discriminative_models.base import BaseDiscModel
from counterfactuals.generative_models.base import BaseGenModel

# Experimenting with custom autograd function
# TODO: Move to separate file
# class OneHotSoftmax(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, temp=0.03):
#         # Store input and temperature for use in backward
#         ctx.save_for_backward(input)
#         ctx.temp = temp

#         # Compute argmax and one-hot encode it
#         indices = input.argmax(dim=1)
#         one_hot = torch.nn.functional.one_hot(
#             indices, num_classes=input.size(1)
#         ).float()

#         return one_hot

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Retrieve saved input and temperature
#         (input,) = ctx.saved_tensors
#         temp = ctx.temp

#         # Compute gradients of softmax with respect to input
#         softmax = torch.nn.functional.softmax(input / temp, dim=1)
#         grad_input = grad_output * (softmax * (1 - softmax) / temp)

#         return grad_input, None  # None corresponds to no gradient for temp


class TorchDequantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1e-6

    def forward(self, x):
        # x[:, 2:] = x[:, 2:] + torch.tensor(0.5)
        x[:, 2:] = x[:, 2:] + torch.sigmoid(torch.randn(x[:, 2:].size()))
        x[:, 2:] = x[:, 2:] / 2
        x[:, 2:] = self.alpha + (1 - 2 * self.alpha) * x[:, 2:]
        result = x.clone()
        new_tensor = torch.log(x[:, 2:] / (torch.tensor(1.0) - x[:, 2:]))
        result[:, 2:] = new_tensor
        return result


torch_dequantizer = TorchDequantizer()


ALPHA = 1e-6


def logit(x):
    x_clone = x.clone()  # Clone to avoid in-place modification issues
    x_clone[:, 2:4] = torch.logit(x[:, 2:4], eps=1e-6)
    return x_clone


class PPCEF(BaseCounterfactual):
    def __init__(
        self,
        gen_model: BaseGenModel,
        disc_model: BaseDiscModel,
        disc_model_criterion,
        device=None,
    ):
        self.disc_model_criterion = disc_model_criterion
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.device = device if device is not None else "cpu"
        self.gen_model.to(self.device)
        self.disc_model.to(self.device)
        self.beta = 0

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
        epoch = search_step_kwargs.get("epoch", None)
        categorical_intervals = search_step_kwargs.get("categorical_intervals", None)

        log_prob_threshold = search_step_kwargs.get("log_prob_threshold", None)
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if log_prob_threshold is None:
            raise ValueError("Parameter 'log_prob_threshold' should be in kwargs")

        dist = torch.linalg.vector_norm(delta, dim=1, ord=2)

        cf = x_origin + delta
        if categorical_intervals:
            tau = 1.0 - 0.99 / self.epochs * epoch
            for interval in categorical_intervals:
                # cf[:, interval] = torch.nn.functional.gumbel_softmax(
                #     cf[:, interval], tau=tau, dim=1
                # )
                cf[:, interval] = torch.nn.functional.softmax(cf[:, interval], dim=1)

        disc_logits = self.disc_model.forward(cf)
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
            x_origin + delta, context=context_target.type(torch.float32)
        )

        max_inner = torch.nn.functional.relu(
            log_prob_threshold * 0.5 - p_x_param_c_target
        )

        # regularization_loss = self.compute_regularization_loss(cf, categorical_intervals)

        loss = dist + alpha * (loss_disc + max_inner)
        return {
            "loss": loss,
            "dist": dist,
            "max_inner": max_inner,
            "loss_disc": loss_disc,
        }

    def compute_regularization_loss(
        self, cf: torch.Tensor, categorical_intervals
    ) -> torch.Tensor:
        # deprecated, categorical_intervals follow different implementation now.
        regularization_loss = 0.0
        for v in categorical_intervals:
            regularization_loss += torch.pow(
                torch.sum((torch.sum(cf[:, v[0] : v[1]], dim=1) - 1.0)), 2
            )

        return regularization_loss

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
        self.epochs = epochs
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

            for epoch in (epoch_pbar := tqdm(range(epochs))):
                search_step_kwargs["epoch"] = epoch
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

                disc_loss = loss_components["loss_disc"].detach().cpu().mean().item()
                prob_loss = loss_components["max_inner"].detach().cpu().mean().item()
                epoch_pbar.set_description(
                    f"Discriminator loss: {disc_loss:.4f}, Prob loss: {prob_loss:.4f}"
                )
                # if disc_loss < patience_eps and prob_loss < patience_eps:
                #     break

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
