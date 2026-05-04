"""
Data ingestion package. Pipeline order (main.py):
1. DataDownloader  — SPY OHLCV from Yahoo Finance (auto-adjusted).
2. FREDLoader      — VIX from FRED; forward-filled; adds _diff/_chg.
3. DataAligner     — inner-join on US equity trading dates.
4. DataLoader      — HDF5 persistence (keys: data_raw, engineered_features).
"""

from .downloader import DataDownloader
from .fred_loader import FREDLoader
from .aligner import DataAligner
from .loader import DataLoader

__all__ = [
    'DataDownloader',
    'FREDLoader',
    'DataAligner',
    'DataLoader',
]
