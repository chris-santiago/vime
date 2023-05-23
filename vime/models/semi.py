import typing as T

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from vime.models.encoder import VimeEncoder
from vime.models.modules import MaskGenerator, PretextGenerator


class VimeLearner(pl.LightningModule):
    """VIME module for semi-supervised learning."""

    def __init__(
        self,
        encoder_ckpt: str,
        classifier: pl.LightningModule,
        n_augments: int = 3,
        beta: float = 0.5,
        optim: T.Optional[
            T.Callable[[torch.optim.Optimizer], torch.optim.Optimizer]
        ] = None,
        score_func: T.Optional[torchmetrics.Metric] = None,
    ):
        super().__init__()
        self.encoder = VimeEncoder.load_from_checkpoint(encoder_ckpt)
        self.classifier = classifier
        self.n_augments = n_augments
        self.beta = beta
        self.optim = optim
        self.get_mask = MaskGenerator(p=self.encoder.hparams.p_mask)
        self.get_pretext = PretextGenerator()
        self.loss_func_supervised = nn.CrossEntropyLoss()
        self.loss_func_consistency = nn.PairwiseDistance()
        self.score_func = score_func

        self.save_hyperparameters()

    def training_step(self, batch, idx):
        x, y = batch
        mask = self.get_mask(x)

        # This is now 3D tensor of augmented original data
        augments = torch.stack(
            [self.get_pretext(x, mask)[0] for _ in range(self.n_augments)], dim=1
        )
        augment_labels = torch.stack([y for _ in range(self.n_augments)], dim=1)

        z_augments = self.encoder.encode(augments)
        preds_augments = self.classifier(z_augments)

        ohe_labels = torch.zeros(*preds_augments.shape, device=self.device)
        ohe_labels.scatter_(2, augment_labels.unsqueeze(-1), 1)

        consistency_loss = self.loss_func_consistency(preds_augments, ohe_labels).mean()

        z = self.encoder.encode(x)
        preds = self.classifier(z)
        supervised_loss = self.loss_func_supervised(preds, y)

        total_loss = supervised_loss * (1 - self.beta) + consistency_loss * self.beta

        self.log(
            "loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )

        metrics = {
            "train-loss-supervised": supervised_loss,
            "train-loss-consistency": consistency_loss,
            "train-loss": total_loss,
        }

        if self.score_func:
            score = self.score_func(preds, y)
            self.log(
                "acc", score, on_step=True, on_epoch=True, prog_bar=True, logger=False
            )
            metrics["train-score"] = score

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch, idx):
        x, y = batch
        mask = self.get_mask(x)

        # This is now 3D tensor of augmented original data
        augments = torch.stack(
            [self.get_pretext(x, mask)[0] for _ in range(self.n_augments)], dim=1
        )
        augment_labels = torch.stack([y for _ in range(self.n_augments)], dim=1)

        z_augments = self.encoder.encode(augments)
        preds_augments = self.classifier(z_augments)

        ohe_labels = torch.zeros(*preds_augments.shape, device=self.device)
        ohe_labels.scatter_(2, augment_labels.unsqueeze(-1), 1)

        consistency_loss = self.loss_func_consistency(preds_augments, ohe_labels).mean()

        z = self.encoder.encode(x)
        preds = self.classifier(z)
        supervised_loss = self.loss_func_supervised(preds, y)

        total_loss = supervised_loss * (1 - self.beta) + consistency_loss * self.beta

        self.log(
            "loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )

        metrics = {
            "valid-loss-supervised": supervised_loss,
            "valid-loss-consistency": consistency_loss,
            "valid-loss": total_loss,
        }

        if self.score_func:
            score = self.score_func(preds, y)
            self.log(
                "acc", score, on_step=True, on_epoch=True, prog_bar=True, logger=False
            )
            metrics["valid-score"] = score

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def configure_optimizers(self):
        if self.optim:
            return self.optim(self.parameters())
        return torch.optim.Adam(self.parameters())
