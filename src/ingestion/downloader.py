"""
Data Downloader — SPY daily OHLCV from Yahoo Finance.

Key decisions:
- auto_adjust=True: splits and dividends are absorbed into all price columns
  so returns are continuous and technical indicators are not distorted.
- Single asset (SPY): avoids cross-sectional noise and alignment issues.
- Start 2004-01-01: provides enough warmup bars for the slowest indicator (SMA200).
- MultiIndex (ticker, date) output: required format for the feature engineering pipeline.
- yfinance column normalisation: three output formats exist across library versions;
  all are handled transparently (see _normalize_columns).
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import yfinance as yf
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
yf.set_tz_cache_location(str(PROJECT_ROOT / "data/cache"))

EXPECTED_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume']

_PRICE_FIELDS = frozenset({'Close', 'High', 'Low', 'Open', 'Volume',
                            'Dividends', 'Stock Splits', 'Capital Gains'})


class DataDownloader:
    """Downloads SPY daily OHLCV from Yahoo Finance as MultiIndex (ticker, date)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.raw_data_path = PROJECT_ROOT / ingestion_config.get('raw_data_path', 'data/raw/')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.start_date = ingestion_config.get('start_date', '2004-01-01')
        self.end_date = ingestion_config.get('end_date')
        self.frequency = ingestion_config.get('frequency', '1d')

        logger.info(f"DataDownloader initialised  |  path: {self.raw_data_path}")
        logger.info(f"Date range: {self.start_date} to {self.end_date or 'today'}")

    def download_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download daily OHLCV for one or more tickers.

        Returns MultiIndex (ticker, date) DataFrame with columns
        ['Close', 'High', 'Low', 'Open', 'Volume'].
        Falls back to Ticker.history() if the batch API returns empty data.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date

        logger.info(f"Downloading {len(tickers)} tickers: {tickers}")
        logger.info(f"Date range: {start} to {end or 'today'}")

        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                interval=self.frequency,
                progress=False,
                auto_adjust=True,
                group_by='ticker'
            )

            if data.empty and len(tickers) == 1:
                data = self._download_single_ticker_fallback(tickers[0], start, end)

            if data.empty:
                raise ValueError(f"No data returned for tickers: {tickers}")

            data = self._normalize_columns(data, tickers)
            stacked = self._stack_to_multiindex(data)

            available = [c for c in EXPECTED_COLUMNS if c in stacked.columns]
            stacked = stacked[available]

            logger.success(
                f"Download complete  |  shape: {stacked.shape}  |  "
                f"tickers: {stacked.index.get_level_values('ticker').unique().tolist()}"
            )
            logger.info(
                f"Date range: {stacked.index.get_level_values('date').min()} to "
                f"{stacked.index.get_level_values('date').max()}"
            )

            return stacked

        except Exception as e:
            logger.error(f"Failed to download {tickers}: {e}")
            raise ValueError(f"Download failed for {tickers}: {e}") from e

    def download_all_assets(self) -> pd.DataFrame:
        """Download every ticker listed in config['data_sources']."""
        logger.info("Starting download of all configured assets")

        data_sources = self.config.get('data_sources', {})
        tickers = [v.get('ticker') for v in data_sources.values() if v.get('ticker')]

        if not tickers:
            raise ValueError("No tickers specified in configuration 'data_sources'")

        logger.info(f"Configured tickers: {tickers}")
        data = self.download_multiple_tickers(tickers)

        for ticker in data.index.get_level_values('ticker').unique():
            ticker_data = data.xs(ticker, level='ticker')
            logger.info(
                f"  {ticker}: {len(ticker_data)} rows  "
                f"({ticker_data.index.min().date()} → {ticker_data.index.max().date()})"
            )

        return data

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return a quality summary dict (row counts, date range, missing values)."""
        tickers = data.index.get_level_values('ticker').unique().tolist()

        summary: Dict[str, Any] = {
            'total_rows': len(data),
            'num_tickers': len(tickers),
            'tickers': tickers,
            'columns': data.columns.tolist(),
            'date_range': {
                'start': str(data.index.get_level_values('date').min()),
                'end': str(data.index.get_level_values('date').max())
            },
            'missing_values': int(data.isnull().sum().sum()),
            'ticker_details': {}
        }

        for ticker in tickers:
            ticker_data = data.xs(ticker, level='ticker')
            summary['ticker_details'][ticker] = {
                'rows': len(ticker_data),
                'start_date': str(ticker_data.index.min()),
                'end_date': str(ticker_data.index.max()),
                'missing_values': int(ticker_data.isnull().sum().sum())
            }

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_single_ticker_fallback(
        self,
        ticker: str,
        start: str,
        end: Optional[str]
    ) -> pd.DataFrame:
        """
        Fallback to Ticker.history() when yf.download() returns empty data.
        Returns a flat-index DataFrame; column normalisation is handled by the caller.
        """
        logger.warning(
            f"yf.download() returned empty data for '{ticker}' — "
            "falling back to Ticker.history()"
        )

        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(
            start=start, end=end, interval=self.frequency, auto_adjust=True
        )

        logger.debug(f"Fallback columns: {data.columns.tolist()}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(c).strip() for c in data.columns]

        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        if data.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}' via both "
                "yf.download() and Ticker.history()"
            )

        logger.success(f"Fallback successful for '{ticker}'  |  {len(data)} rows")
        return data

    def _normalize_columns(self, data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Normalise yfinance column format to (ticker, field) MultiIndex.

        Three formats exist across library versions:
          <0.2.50 single ticker : flat ['Close', 'High', ...]
          <0.2.50 multi-ticker  : MultiIndex (ticker, field)  level 0 = ticker
          ≥0.2.61 any           : MultiIndex (field, ticker)  level 0 = field  ← swapped

        All are converted to (ticker, field) before stacking.
        """
        if isinstance(data.columns, pd.MultiIndex):
            l0 = set(data.columns.get_level_values(0).unique())
            if l0.issubset(_PRICE_FIELDS):
                # New format: level 0 is field — swap to put ticker first
                data.columns = data.columns.swaplevel(0, 1)
        else:
            if len(tickers) == 1:
                data.columns = pd.MultiIndex.from_product(
                    [[tickers[0]], data.columns],
                    names=['ticker', 'field']
                )

        return data

    def _stack_to_multiindex(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape from wide (date × (ticker, field)) to long (ticker, date) × field.
        Strips timezone from the date level to match the tz-naive FRED index.
        """
        try:
            stacked = data.stack(level=0, future_stack=True)
        except TypeError:
            stacked = data.stack(level=0)

        stacked = stacked.swaplevel(0, 1).sort_index()
        stacked.index.names = ['ticker', 'date']

        date_level = stacked.index.levels[1]
        if date_level.tz is not None:
            stacked.index = stacked.index.set_levels(
                date_level.tz_convert(None), level='date'
            )

        return stacked
