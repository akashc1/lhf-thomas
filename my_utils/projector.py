import torch
from torch import nn


class Project2Range(nn.Module):
    r"""Project the input to a range."""

    def __init__(self, min: int, max: int) -> None:
        r"""Initialize the module.

        Args:
            min: The minimum value of the range
            max: The maximum value of the range
        """
        super().__init__()
        self.min = min
        self.max = max
        self.range = self.max - self.min

    def forward(self, x):
        r"""Project the last dimension of the input to the range.

        Args:
            x: The (batch) input tensor.

        Returns:
            The projected tensor with the same dimention as the input.
        """
        return torch.sigmoid(x) * self.range + self.min
