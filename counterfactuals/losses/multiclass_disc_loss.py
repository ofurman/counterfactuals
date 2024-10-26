import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MulticlassDiscLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", eps=0.02
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.type() != torch.LongTensor:
            target = target.long()
        target_mask = torch.eye(input.shape[-1])[target]
        target_mask = target_mask.squeeze(1)  # label 2 one-hot conversion
        non_target_mask = (~target_mask.bool()).float()
        p_target = torch.sum(input * target_mask, dim=1)
        p_max_non_target = torch.max(input * non_target_mask, dim=1).values
        loss = F.relu(p_max_non_target + self.eps - p_target)
        return loss
