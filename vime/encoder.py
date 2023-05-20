from functools import partial

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


class VimeEncoder(L.LightningModule):
    def __init__(self, neurons=256, p_mask=0.25, optim=None):
        super().__init__()
        self.optim = optim
        self.get_mask = MaskGenerator(p=p_mask)
        self.get_pretext = PretextGenerator()
        self.loss_func_feature = nn.MSELoss()
        self.loss_func_mask = nn.BCELoss()

        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.LazyLinear(out_features=neurons),
            nn.ReLU(),
        )

        self.feature_estimator = nn.Sequential(
            nn.Linear(in_features=neurons, out_features=784),
            nn.Sigmoid(),
        )

        self.mask_estimator = nn.Sequential(
            nn.Linear(in_features=neurons, out_features=784),
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
        total_loss = loss_feature + loss_mask
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


if __name__ == "__main__":
    from pytorch_lightning.callbacks.progress import RichProgressBar

    import vime.data
    import vime.utils

    BATCH = 128
    NUM_WORKERS = 10
    EPOCHS = 10

    device = vime.utils.set_device()

    train, test = vime.data.get_mnist_datasets()
    train_dl = DataLoader(
        train, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS
    )
    test_dl = DataLoader(test, batch_size=BATCH, num_workers=NUM_WORKERS)

    optim = partial(torch.optim.RMSprop, lr=0.01)
    model = VimeEncoder(neurons=256, p_mask=0.33, optim=optim)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="mps",
        devices=1,
        callbacks=[RichProgressBar(refresh_rate=5, leave=True)],
    )
    trainer.fit(model=model, train_dataloaders=train_dl)
