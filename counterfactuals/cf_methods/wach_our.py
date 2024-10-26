import torch

from counterfactuals.cf_methods.ppcef.ppcef_base import BaseCounterfactualModel


class WACH(BaseCounterfactualModel):
    def __init__(
        self, gen_model, disc_model, disc_model_criterion, device=None, neptune_run=None
    ):
        self.disc_model_criterion = disc_model_criterion
        super().__init__(gen_model, disc_model, device, neptune_run)

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
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if delta is None:
            raise ValueError("Parameter 'delta' should be in kwargs")

        dist = torch.linalg.norm(x_origin - x_param, axis=1, ord=2)

        disc_logits = self.disc_model.forward(x_param)
        disc_logits = (
            disc_logits.reshape(-1) if disc_logits.shape[0] == 1 else disc_logits
        )
        context_target = (
            context_target.reshape(-1)
            if context_target.shape[0] == 1
            else context_target
        )
        loss_disc = self.disc_model_criterion(disc_logits, context_target)

        loss = dist + alpha * (loss_disc)
        return {
            "loss": loss,
            "dist": dist,
            "loss_disc": loss_disc,
        }
