import hydra
import pytorch_lightning as L
from encoder import VimeEncoder
from pytorch_lightning.callbacks.progress import RichProgressBar
from torch.utils.data import DataLoader

import vime.data


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    train, test = vime.data.get_mnist_datasets()
    train_dl = DataLoader(train, **cfg.dataloader.train)
    DataLoader(test, **cfg.dataloader.test)

    optim = hydra.utils.instantiate(cfg.optimizer.encoder, lr=cfg.encoder.lr)
    model = VimeEncoder(
        hidden_size=cfg.encoder.hidden_size,
        encoder_layers=cfg.encoder.encoder_layers,
        pretext_layers=cfg.encoder.pretext_layers,
        p_mask=cfg.encoder.p_mask,
        alpha=cfg.encoder.alpha,
        optim=optim,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[RichProgressBar(refresh_rate=5, leave=True)],
    )
    trainer.fit(model=model, train_dataloaders=train_dl)


if __name__ == "__main__":
    main()
