import typing as T

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from vime.models.modules import LinearLayer


class VimeMLP(pl.LightningModule):
    """VIME MLP module for classification."""

    def __init__(
        self,
        hidden_size: int = 128,
        n_layers: int = 3,
        out_size: int = 10,
        optim: T.Optional[
            T.Callable[[torch.optim.Optimizer], torch.optim.Optimizer]
        ] = None,
        score_func: T.Optional[torchmetrics.Metric] = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.optim = optim
        self.loss_func = nn.CrossEntropyLoss()
        self.score_func = (
            score_func
            if score_func
            else torchmetrics.Accuracy(task="multiclass", num_classes=10)
        )

        self.layers = nn.ModuleList(
            [
                LinearLayer(hidden_size, hidden_size, batch_norm)
                for i in range(n_layers - 2)
            ]
        )

        self.model = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size),
            *self.layers,
            nn.Linear(in_features=hidden_size, out_features=out_size),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log(
            "train-loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        self.log(
            "valid-loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        if self.optim:
            return self.optim(self.parameters())
        return torch.optim.Adam(self.parameters())
