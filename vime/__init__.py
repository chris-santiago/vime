"""Package initialization."""
from importlib.metadata import version

from _constants import Constants
from models.encoder import VimeEncoder
from models.mlp import VimeMLP

from data import MnistData

__version__ = version("vime")
__all__ = [Constants, VimeMLP, VimeEncoder, MnistData]
