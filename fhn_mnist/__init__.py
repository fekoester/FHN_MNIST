"""FitzHugh–Nagumo networks on MNIST: backprop through time and three-factor plasticity."""

from .backprop import MLP, FHNClassifier, evaluate
from .dynamics import FHNParams, fhn_step, simulate
from .plasticity import FHNBioplasticNet, PlasticityCfg, evaluate_bio

__all__ = ["FHNClassifier", "MLP", "evaluate", "FHNParams", "fhn_step", "simulate", "FHNBioplasticNet",
           "PlasticityCfg", "evaluate_bio"]
__version__ = "0.2.0"
