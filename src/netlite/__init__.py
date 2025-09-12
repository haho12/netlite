# make imports from layers.py etc available in the package name 'netlite',
# e.g. allow
# import netlite as nl
# l1 = nl.ReLU()

from .layers import *
from .loss_functions import *
from .neural_network import *
from .optimizer import *

from .dataloader_mnist import *
from .dataloader_cifar10 import *

# set nl.__version__
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("netlite") # get version from pyproject.toml
except PackageNotFoundError:
    __version__ = "0.0.0"            # fallback if not installed
