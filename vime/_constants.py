import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    SEED = 42
    HERE = pathlib.Path(__file__)
    SRC = HERE.parents[0]
    REPO = HERE.parents[1]
    DATA = REPO.joinpath("data")
