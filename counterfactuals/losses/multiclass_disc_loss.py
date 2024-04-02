import torch


class MulticlassDiscLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", eps=0.02
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        one_hot = torch.eye(input.shape[-1])[target].squeeze(
            1
        )  # label 2 one-hot conversion
        dot_product = torch.einsum(
            "nc,nc->n", one_hot, input
        )  # n - batch size, c - number of classes
        loss = dot_product - torch.max(input, dim=1).values - self.eps
        loss = torch.linalg.norm(loss.view(-1, 1), ord=1, dim=1)
        return loss
