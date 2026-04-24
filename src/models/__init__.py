"""
Models package.

Provides model training, walk-forward cross-validation, backtesting,
and regime detection.
"""

from .train import ModelTrainer
from .walk_forward import WalkForwardCV
from .backtest import Backtester
from .regime_detection import RegimeDetector

__all__ = [
    'ModelTrainer',
    'WalkForwardCV',
    'Backtester',
    'RegimeDetector'
]
