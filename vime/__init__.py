"""Package initialization."""
from importlib.metadata import version

from ._constants import Constants
from .data import get_mnist_test, get_mnist_train
from .models.encoder import VimeEncoder
from .models.mlp import VimeMLP
from .models.semi import VimeLearner
from .utils import instantiate_callbacks

__version__ = version("vime")
__all__ = [
    Constants,
    VimeMLP,
    VimeEncoder,
    VimeLearner,
    get_mnist_test,
    get_mnist_train,
    instantiate_callbacks,
]
