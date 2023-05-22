import hydra

import vime.utils


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    data = hydra.utils.instantiate(cfg.dataloader)
    optim = hydra.utils.instantiate(cfg.model.optimizer)
    model = hydra.utils.instantiate(cfg.model.nn, optim=optim)
    callbacks = vime.utils.instantiate_callbacks(cfg.callbacks)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=data)
    trainer.checkpoint_callback.to_yaml()


if __name__ == "__main__":
    main()
