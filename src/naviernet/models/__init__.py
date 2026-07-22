"""Network architectures."""

from naviernet.models.layers import AdaptiveTanh, FourierFeatures
from naviernet.models.pinn import BubblePINN, FieldNet

__all__ = ["AdaptiveTanh", "BubblePINN", "FieldNet", "FourierFeatures"]
