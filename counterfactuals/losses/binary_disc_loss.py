import torch


class BinaryDiscLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", eps=0.05
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Alternative implementation, have slightly different results TODO: check in future
        # return F.relu(torch.linalg.vector_norm(input - target) - 0.5 + self.eps)
        scaled = -2 * target + 1
        return torch.nn.functional.relu(
            scaled * (input - torch.Tensor([0.5]) + scaled * torch.Tensor([self.eps]))
        )
