import torch
from torch import nn as nn


class MaskGenerator(nn.Module):
    """Module for generating Bernoulli mask."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor):
        """Generate Bernoulli mask."""
        p_mat = torch.ones_like(x) * self.p
        return torch.bernoulli(p_mat)


class PretextGenerator(nn.Module):
    """Module for generating training pretext."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def shuffle(x: torch.tensor):
        """Shuffle each column in a tensor."""
        m, n = x.shape
        x_bar = torch.zeros_like(x)
        for i in range(n):
            idx = torch.randperm(m)
            x_bar[:, i] += x[idx, i]
        return x_bar

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """Generate corrupted features and corresponding mask."""
        shuffled = self.shuffle(x)
        corrupt_x = x * (1.0 - mask) + shuffled * mask
        corrupt_mask = 1.0 * (x != corrupt_x)  # ensure float type
        return corrupt_x, corrupt_mask


class LinearLayer(nn.Module):
    """
    Module to create a sequential block consisting of:

        1. Linear layer
        2. (optional) Batch normalization layer
        3. ReLu activation layer
    """

    def __init__(self, input_size: int, output_size: int, batch_norm: bool = False):
        super().__init__()
        self.size_in = input_size
        self.size_out = output_size
        if batch_norm:
            self.model = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
            )
        else:
            self.model = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

    def forward(self, x: torch.tensor):
        """Run inputs through linear block."""
        return self.model(x)
