"""
Data Ingestion Package
======================

Pipeline steps covered by this package (executed in order by main.py):

    1. DataDownloader  — downloads SPY OHLCV from Yahoo Finance (auto-adjusted).
    2. FREDLoader      — downloads VIX from FRED; forward-fills; adds _diff/_chg.
    3. DataAligner     — inner-joins yfinance and FRED on US equity trading dates.
    4. DataLoader      — persists/loads DataFrames to/from HDF5 (two keys:
                         ``data_raw`` and ``engineered_features``).
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
