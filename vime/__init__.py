"""Package initialization."""
from importlib.metadata import version

from _constants import Constants
from models.encoder import VimeEncoder
from models.mlp import VimeMLP
from models.semi import VimeLearner

from data import get_mnist_test, get_mnist_train

__version__ = version("vime")
__all__ = [
    Constants,
    VimeMLP,
    VimeEncoder,
    VimeLearner,
    get_mnist_test,
    get_mnist_train,
]
