"""Package initialization."""
import pathlib
from dataclasses import dataclass
from importlib.metadata import version

__version__ = version("vime")


@dataclass(frozen=True)
class Constants:
    HERE = pathlib.Path(__file__)
    SRC = HERE.parents[0]
    REPO = HERE.parents[1]
    DATA = REPO.joinpath("data")
