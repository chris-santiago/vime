import hydra
import pytorch_lightning as pl
from encoder import VimeEncoder
from pytorch_lightning.callbacks.progress import RichProgressBar


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    data = hydra.utils.instantiate(cfg.dataloader)
    optim = hydra.utils.instantiate(cfg.optimizer.encoder)

    model = VimeEncoder(
        hidden_size=cfg.encoder.hidden_size,
        encoder_layers=cfg.encoder.encoder_layers,
        pretext_layers=cfg.encoder.pretext_layers,
        p_mask=cfg.encoder.p_mask,
        alpha=cfg.encoder.alpha,
        optim=optim,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[RichProgressBar(refresh_rate=5, leave=True)],
    )
    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
