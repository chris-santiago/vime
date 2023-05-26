import hydra

import vime.data
import vime.utils


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    train_dl = hydra.utils.instantiate(cfg.data.unlabeled)
    optim = hydra.utils.instantiate(cfg.model.optimizer)
    model = hydra.utils.instantiate(cfg.model.nn, optim=optim)
    callbacks = vime.utils.instantiate_callbacks(cfg.callbacks)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_dl)
    trainer.checkpoint_callback.to_yaml()
    # TODO Consider adding manual checkpoint save to static directory for
    # TODO full training pipeline use: train-full = train-sef => train-semi


if __name__ == "__main__":
    main()
