import torch
from counterfactuals.cf_methods.base import BaseCounterfactualModel


class OneHotSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, temp=0.03):
        # Store input and temperature for use in backward
        ctx.save_for_backward(input)
        ctx.temp = temp

        # Compute argmax and one-hot encode it
        indices = input.argmax(dim=1)
        one_hot = torch.nn.functional.one_hot(
            indices, num_classes=input.size(1)
        ).float()

        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input and temperature
        (input,) = ctx.saved_tensors
        temp = ctx.temp

        # Compute gradients of softmax with respect to input
        softmax = torch.nn.functional.softmax(input / temp, dim=1)
        grad_input = grad_output * (softmax * (1 - softmax) / temp)

        return grad_input, None  # None corresponds to no gradient for temp


class PPCEF(BaseCounterfactualModel):
    def __init__(
        self,
        gen_model,
        disc_model,
        disc_model_criterion,
        device=None,
        neptune_run=None,
    ):
        self.disc_model_criterion = disc_model_criterion
        self.one_hot_softmax = OneHotSoftmax.apply
        super().__init__(gen_model, disc_model, device, neptune_run)

    def _round_categorical_features(self, x_param, categorical_features_list):
        for i, feature in enumerate(categorical_features_list):
            x_param[:, feature] = torch.round(x_param[:, feature])
        return x_param

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
        categorical_features_lists = search_step_kwargs.get(
            "categorical_features_lists", None
        )
        if alpha is None:
            raise ValueError("Parameter 'alpha' should be in kwargs")
        if delta is None:
            raise ValueError("Parameter 'delta' should be in kwargs")

        if categorical_features_lists:
            # x_param = self.one_hot_softmax(x_param)
            new_tensor = torch.zeros_like(x_param)
            first_cat_feature = categorical_features_lists[0][0]
            new_tensor[:, :first_cat_feature] = x_param[:, :first_cat_feature]
            for feature in categorical_features_lists:
                new_tensor[:, feature] = self.one_hot_softmax(x_param[:, feature])
                # new_tensor[:, feature] = F.softmax(x_param[:, feature] / 0.1, dim=1)

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

        loss = dist + alpha * (max_inner + loss_disc)
        return {
            "loss": loss,
            "dist": dist,
            "max_inner": max_inner,
            "loss_disc": loss_disc,
        }
