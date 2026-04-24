"""
Data Ingestion Package

Provides classes for downloading, loading, and aligning financial data.
"""

from .downloader import DataDownloader
from .fred_loader import FREDLoader
from .aligner import DataAligner
from .loader import DataLoader

__all__ = [
    'DataDownloader',
    'FREDLoader',
    'DataAligner',
    'DataLoader'
]
