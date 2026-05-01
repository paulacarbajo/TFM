"""
Models package.

Provides model training, walk-forward cross-validation, and regime detection.
"""

from .train import ModelTrainer
from .walk_forward import WalkForwardCV
from .regime_detection import RegimeDetector

__all__ = [
    'ModelTrainer',
    'WalkForwardCV',
    'RegimeDetector'
]
