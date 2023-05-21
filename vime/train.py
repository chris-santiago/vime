import hydra


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    data = hydra.utils.instantiate(cfg.dataloader)
    optim = hydra.utils.instantiate(cfg.optimizer)
    model = hydra.utils.instantiate(cfg.model, optim=optim)
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
