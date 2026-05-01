"""
Features Package
================

Pipeline steps covered by this package (executed in order by main.py):

    5. FeatureEngineer      — computes 11 stationary technical features
                              (ret_5d, ret_21d, vol_20d, rsi_14, macd_line,
                              macd_signal, macd_hist, bb_pct, bb_width,
                              atr_14, volume_ratio) plus auxiliary columns
                              (ret_1d_forward, ret_10d_forward, legacy labels).
    6. TripleBarrierLabeler — adds label (ternary) and label_binary (binary)
                              using dynamic volatility-scaled barriers
                              (k=1.0, max_holding=8 days).
"""

from .feature_engineering import FeatureEngineer
from .triple_barrier import TripleBarrierLabeler

__all__ = [
    'FeatureEngineer',
    'TripleBarrierLabeler',
]
