import pytorch_lightning as pl
import torch
import torch.nn as nn


class MaskGenerator(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        p_mat = torch.ones_like(x) * self.p
        return torch.bernoulli(p_mat)


class PretextGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def shuffle(x):
        m, n = x.shape
        x_bar = torch.zeros_like(x)
        for i in range(n):
            idx = torch.randperm(m)
            x_bar[:, i] += x[idx, i]
        return x_bar

    def forward(self, x, mask):
        shuffled = self.shuffle(x)
        corrupt_x = x * (1.0 - mask) + shuffled * mask
        corrupt_mask = 1.0 * (x != corrupt_x)  # ensure float type
        return corrupt_x, corrupt_mask


class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, batch_norm=False):
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

    def forward(self, x):
        return self.model(x)


class VimeEncoder(pl.LightningModule):
    def __init__(
        self,
        hidden_size=256,
        encoder_layers=2,
        pretext_layers=2,
        out_size=784,
        p_mask=0.25,
        alpha=0.5,
        optim=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.optim = optim
        self.get_mask = MaskGenerator(p=p_mask)
        self.get_pretext = PretextGenerator()
        self.loss_func_feature = nn.MSELoss()
        self.loss_func_mask = nn.BCELoss()

        self.save_hyperparameters()

        self.encoder_layers = nn.ModuleList(
            [LinearLayer(hidden_size, hidden_size) for i in range(encoder_layers - 1)]
        )

        self.feature_layers = nn.ModuleList(
            [LinearLayer(hidden_size, hidden_size) for i in range(pretext_layers - 1)]
        )

        self.mask_layers = nn.ModuleList(
            [LinearLayer(hidden_size, hidden_size) for i in range(pretext_layers - 1)]
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
        total_loss = self.alpha * loss_feature + (1 - self.alpha) * loss_mask
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return total_loss

    def configure_optimizers(self):
        if self.optim:
            return self.optim(self.parameters())
        return torch.optim.Adam(self.parameters())
