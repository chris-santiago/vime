import hydra

import vime.data
import vime.utils


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    train_dl = hydra.utils.instantiate(cfg.data.labeled)
    valid_dl = hydra.utils.instantiate(cfg.data.valid)
    optim = hydra.utils.instantiate(cfg.model.optimizer)
    model = hydra.utils.instantiate(cfg.model.nn, optim=optim)
    callbacks = vime.utils.instantiate_callbacks(cfg.callbacks)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )
    trainer.checkpoint_callback.to_yaml()


if __name__ == "__main__":
    main()
