import json

import hydra
from sklearn.metrics import accuracy_score

import vime

constants = vime.Constants()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    train = hydra.utils.instantiate(cfg.data.base_train)
    test = hydra.utils.instantiate(cfg.data.base_test)

    x_train, y_train = train.tensors[0].numpy(), train.tensors[1].numpy()
    x_test, y_test = test.tensors[0].numpy(), test.tensors[1].numpy()

    mod = hydra.utils.instantiate(cfg.model.estimator)
    mod.fit(x_train, y_train)

    preds = mod.predict(x_test)
    score = accuracy_score(y_test, preds)
    results = {
        "model": cfg.model.name,
        "n_labeled": cfg.data.n_labeled,
        "score": score,
    }

    filepath = constants.REPO.joinpath("outputs", f"{cfg.data.name}-baselines.txt")
    with open(filepath, "a") as fp:
        fp.write(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
