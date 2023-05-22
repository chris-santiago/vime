import typing as T

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


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


class VimeEncoder(pl.LightningModule):
    """VIME Encoder module for self-supervised learning."""

    def __init__(
        self,
        hidden_size: int = 256,
        encoder_layers: int = 2,
        pretext_layers: int = 2,
        out_size: int = 784,
        p_mask: float = 0.25,
        alpha: float = 0.5,
        optim: T.Optional[
            T.Callable[[torch.optim.Optimizer], torch.optim.Optimizer]
        ] = None,
        score_func: T.Optional[torchmetrics.Metric] = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.optim = optim
        self.get_mask = MaskGenerator(p=p_mask)
        self.get_pretext = PretextGenerator()
        self.loss_func_feature = nn.MSELoss()
        self.loss_func_mask = nn.BCELoss()
        self.score_func = score_func

        self.save_hyperparameters()

        self.encoder_layers = nn.ModuleList(
            [
                LinearLayer(hidden_size, hidden_size, batch_norm)
                for i in range(encoder_layers - 1)
            ]
        )

        self.feature_layers = nn.ModuleList(
            [
                LinearLayer(hidden_size, hidden_size, batch_norm)
                for i in range(pretext_layers - 1)
            ]
        )

        self.mask_layers = nn.ModuleList(
            [
                LinearLayer(hidden_size, hidden_size, batch_norm)
                for i in range(pretext_layers - 1)
            ]
        )

        self.encoder = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size),
            *self.encoder_layers,
            nn.ReLU(),
        )

        self.feature_estimator = nn.Sequential(
            *self.feature_layers,
            nn.Linear(in_features=hidden_size, out_features=out_size),
            nn.Sigmoid(),
        )

        self.mask_estimator = nn.Sequential(
            *self.mask_layers,
            nn.Linear(in_features=hidden_size, out_features=out_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.feature_estimator(x), self.mask_estimator(x)

    def training_step(self, batch, idx):
        x, y = batch
        mask = self.get_mask(x)
        corrupt_feature, corrupt_mask = self.get_pretext(x, mask)
        logits_feature, logits_mask = self(corrupt_feature)

        loss_feature = self.loss_func_feature(logits_feature, x)
        # Note that we calculate loss based on corrupted mask vice original mask because the
        # corrupted mask aligns with corrupted features after shuffling.
        loss_mask = self.loss_func_mask(logits_mask, corrupt_mask)
        # Note that we implement alpha as a mixing parameter slightly differently that the original
        # version. The original version applied alpha only to the feature loss, leaving the
        # mask loss constant.
        total_loss = self.alpha * loss_feature + (1 - self.alpha) * loss_mask

        self.log(
            "loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )

        metrics = {
            "train-loss-feature": loss_feature,
            "train-loss-mask": loss_mask,
            "train-loss": total_loss,
        }

        if self.score_func:
            score = self.score_func(logits_feature, x)
            metrics["train-score"] = score

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return total_loss

    def configure_optimizers(self):
        if self.optim:
            return self.optim(self.parameters())
        return torch.optim.Adam(self.parameters())
