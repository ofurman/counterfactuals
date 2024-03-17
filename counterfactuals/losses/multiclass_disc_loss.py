import torch


class MulticlassDiscLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", eps=0.02
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        one_hot = torch.eye(3)[target][:,0,:]
        dot_product = torch.sum(input*one_hot, dim=1)
        return torch.mean(dot_product - torch.max(input, dim=1).values)