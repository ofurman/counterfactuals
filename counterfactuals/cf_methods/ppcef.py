import torch
from counterfactuals.cf_methods.ppcef_base import BasePPCEF


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


class PPCEF(BasePPCEF):
    def __init__(
        self,
        gen_model,
        disc_model,
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
        log_prob_threshold = search_step_kwargs.get("log_prob_threshold", None)
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if log_prob_threshold is None:
            raise ValueError("Parameter 'log_prob_threshold' should be in kwargs")

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

        p_x_param_c_target = self.gen_model(
            x_origin + delta, context=context_target.type(torch.float32)
        )
        max_inner = torch.nn.functional.relu(log_prob_threshold - p_x_param_c_target)

        loss = dist + alpha * (max_inner + loss_disc)
        return {
            "loss": loss,
            "dist": dist,
            "max_inner": max_inner,
            "loss_disc": loss_disc,
        }
