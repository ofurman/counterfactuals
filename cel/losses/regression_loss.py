import torch


class ThresholdRegressionLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        mode="rmse",
        reduction: str = "mean",
        eps=0.02,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.eps = eps
        self.mode = mode

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "ge":
            # input >= target
            loss = torch.nn.functional.relu(input - target)
        elif self.mode == "le":
            # input <= target
            loss = torch.nn.functional.relu(target - input)
        elif self.mode == "mse":
            # Mean Squared Error
            loss = torch.nn.functional.mse_loss(input, target)
        elif self.mode == "rmse":
            loss = torch.nn.functional.mse_loss(input, target)
        elif self.mode == "threshold":
            loss = torch.nn.functional.relu(torch.abs(input - target) - self.eps)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return loss


class PotentialRegressionLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        mode: str = "ge",
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        eps=0.02,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.mode = mode
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "ge":
            # input >= target
            loss = torch.nn.functional.relu(input - target)
        elif self.mode == "le":
            # input <= target
            loss = torch.nn.functional.relu(target - input)
        elif self.mode == "mse":
            # Mean Squared Error
            loss = torch.nn.functional.mse_loss(input, target)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        return loss
