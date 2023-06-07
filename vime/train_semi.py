import json

import hydra

import vime

constants = vime.Constants()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    train_dl = hydra.utils.instantiate(cfg.data.labeled)
    valid_dl = hydra.utils.instantiate(cfg.data.valid)
    optim = hydra.utils.instantiate(cfg.model.optimizer)
    model = hydra.utils.instantiate(cfg.model.nn, optim=optim)
    callbacks = vime.instantiate_callbacks(cfg.callbacks)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )
    trainer.checkpoint_callback.to_yaml()

    results = {
        "model": cfg.model.name,
        "n_labeled": cfg.data.n_labeled,
        "score": trainer.logged_metrics["valid-score"].item(),
    }

    filepath = constants.REPO.joinpath("outputs", f"{cfg.data.name}-baselines.txt")
    with open(filepath, "a") as fp:
        fp.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
