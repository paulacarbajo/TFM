"""
Data Downloader Module
======================

Downloads SPY (S&P 500 ETF) daily price data from Yahoo Finance for the period
2004-2024 and structures it for the feature engineering pipeline.

Design decisions that affect data quality and downstream results
----------------------------------------------------------------
1. **auto_adjust=True** — adjusts all OHLCV values for stock splits and
   dividend distributions. Without adjustment, a 2:1 split cuts the historical
   Close in half, producing artificial −50 % returns that distort both technical
   indicators (RSI, MACD, Bollinger Bands) and triple-barrier labels.
   All 2004-2024 features and labels are computed on these adjusted prices.

2. **Single asset (SPY)** — the model is trained exclusively on SPY to avoid
   cross-sectional noise, data alignment issues, and look-ahead bias that arise
   when mixing assets with different listing histories. Using only one asset
   keeps the signal-to-noise ratio high and the feature set focused.

3. **Start date 2004-01-01** — provides ≥200 trading days before the first
   usable label (2004 Q3), which is the warmup needed for the slowest indicator
   used (SMA200). Faster indicators (ATR14, RSI14, vol_20d) need fewer bars.

4. **MultiIndex (ticker, date)** output format — required by the feature
   engineering pipeline. The ticker level is kept even for a single asset so
   that all downstream code is consistent and can be extended without changes.

5. **yfinance version compatibility** — handles three column formats produced
   by different yfinance releases (see _normalize_columns for details). This
   prevents silent data corruption when the library is updated.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import yfinance as yf
from loguru import logger

# Project root — two levels up from src/ingestion/downloader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set yfinance cache location relative to project root to avoid permission issues
yf.set_tz_cache_location(str(PROJECT_ROOT / "data/cache"))

# OHLCV columns kept after download; others (Dividends, Stock Splits, Capital Gains)
# are produced by yfinance with auto_adjust=True but are not needed downstream.
EXPECTED_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume']

# Price field names as returned by different yfinance versions.
# Used to detect which MultiIndex level contains field names vs ticker names.
_PRICE_FIELDS = frozenset({'Close', 'High', 'Low', 'Open', 'Volume',
                            'Dividends', 'Stock Splits', 'Capital Gains'})


class DataDownloader:
    """
    Downloads SPY daily OHLCV data from Yahoo Finance and returns a
    MultiIndex (ticker, date) DataFrame compatible with the pipeline.

    The download handles three yfinance column formats transparently and
    includes a single-ticker fallback when the batch API returns empty data.
    All prices are split- and dividend-adjusted (auto_adjust=True).

    Attributes:
        config (Dict[str, Any]): Full configuration dictionary.
        raw_data_path (Path): Directory for raw data files.
        start_date (str): Download start date (YYYY-MM-DD).
        end_date (Optional[str]): Download end date; None = today.
        frequency (str): Bar frequency ('1d' for daily).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise DataDownloader from the project configuration.

        Args:
            config: Configuration dictionary. Reads the 'ingestion' section
                    for raw_data_path, start_date, end_date, and frequency.
        """
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.raw_data_path = PROJECT_ROOT / ingestion_config.get('raw_data_path', 'data/raw/')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.start_date = ingestion_config.get('start_date', '2004-01-01')
        self.end_date = ingestion_config.get('end_date')
        self.frequency = ingestion_config.get('frequency', '1d')

        logger.info(f"DataDownloader initialised  |  path: {self.raw_data_path}")
        logger.info(f"Date range: {self.start_date} to {self.end_date or 'today'}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download daily OHLCV data for one or more tickers.

        Adjustment strategy
        ~~~~~~~~~~~~~~~~~~~
        ``auto_adjust=True`` applies split and dividend corrections to all
        price columns so that every bar reflects what an investor holding
        through that event would have seen. Concretely:

        - **Splits**: a 2:1 split on day t halves all historical Close/High/
          Low/Open values so the return series is continuous.
        - **Dividends**: the ex-dividend price drop is absorbed into the
          adjusted Close, so dividend yield is included in returns.

        This is the standard choice for total-return technical analysis and
        ensures that ATR, MACD, Bollinger Bands, and triple-barrier labels
        are not contaminated by corporate-action artefacts.

        Column-format normalisation
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        yfinance has changed its output format across releases.  This method
        normalises all three variants to a canonical (ticker, field) MultiIndex
        before reshaping.  See ``_normalize_columns`` for details.

        Args:
            tickers: List of ticker symbols, e.g. ['SPY'].
            start_date: Override download start date (YYYY-MM-DD).
            end_date: Override download end date (YYYY-MM-DD).

        Returns:
            DataFrame with MultiIndex (ticker, date) and columns
            ['Close', 'High', 'Low', 'Open', 'Volume'].

        Raises:
            ValueError: If all download attempts fail or return empty data.
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

            # Batch API sometimes returns empty for a single ticker due to a
            # known yfinance quirk.  Fall back to the single-ticker API which
            # is more reliable in that case.
            if data.empty and len(tickers) == 1:
                data = self._download_single_ticker_fallback(tickers[0], start, end)

            if data.empty:
                raise ValueError(f"No data returned for tickers: {tickers}")

            # Normalise column format across yfinance versions, then reshape
            # from wide (date × (ticker, field)) to long (ticker, date) × field.
            data = self._normalize_columns(data, tickers)
            stacked = self._stack_to_multiindex(data)

            # Keep only the five OHLCV columns used downstream.
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
        """
        Download every ticker listed in the 'data_sources' configuration section.

        Returns:
            DataFrame with MultiIndex (ticker, date).

        Raises:
            ValueError: If no tickers are configured.
        """
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
        """
        Compute a quality summary for downloaded data.

        Useful for logging and sanity-checking before feature engineering.

        Args:
            data: DataFrame with MultiIndex (ticker, date).

        Returns:
            Dictionary with total rows, date range, missing value counts,
            and per-ticker details.
        """
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
        Download data via yfinance.Ticker.history() when the batch API fails.

        The batch API (yf.download) occasionally returns empty data for a
        single ticker — usually a transient network issue or a change in
        yfinance's internal routing.  The per-ticker .history() call is
        more reliable in those cases.

        The returned DataFrame has a flat DatetimeIndex (not a MultiIndex),
        so it requires column normalisation before stacking, which is handled
        by the caller via _normalize_columns / _stack_to_multiindex.

        Args:
            ticker: Ticker symbol (e.g. 'SPY').
            start: Start date string.
            end: End date string or None.

        Returns:
            Wide DataFrame with a flat DatetimeIndex, timezone stripped.

        Raises:
            ValueError: If this method also returns empty data.
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

        # Flatten MultiIndex columns if the history call produced them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(c).strip() for c in data.columns]

        # Strip timezone so the index is compatible with the FRED DatetimeIndex
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

        yfinance has changed its output format across releases, producing three
        distinct structures that must all be handled:

        +-------------------+--------------------------------------------------+
        | yfinance version  | Column format                                    |
        +===================+==================================================+
        | <0.2.50,          | Flat: ['Close', 'High', 'Low', 'Open', 'Volume'] |
        | single ticker     |                                                  |
        +-------------------+--------------------------------------------------+
        | <0.2.50,          | MultiIndex: [('SPY','Close'), ('SPY','High'), …]  |
        | multi-ticker      | names=['Ticker', 'Price']                        |
        +-------------------+--------------------------------------------------+
        | ≥0.2.61, any      | MultiIndex: [('Close','SPY'), ('High','SPY'), …]  |
        |                   | names=['Price', 'Ticker']  ← levels are swapped  |
        +-------------------+--------------------------------------------------+

        Target (before stacking): ('ticker', 'field') MultiIndex, i.e. level 0
        is the ticker symbol and level 1 is the field name.  This lets
        ``data.stack(level=0)`` pivot tickers into the row index.

        Args:
            data: Raw DataFrame from yf.download() or Ticker.history().
            tickers: List of ticker symbols used in the download request.

        Returns:
            DataFrame with (ticker, field) MultiIndex columns.
        """
        if isinstance(data.columns, pd.MultiIndex):
            l0 = set(data.columns.get_level_values(0).unique())
            if l0.issubset(_PRICE_FIELDS):
                # New format (≥0.2.61): level 0 = field, level 1 = ticker.
                # Swap so level 0 = ticker, level 1 = field.
                data.columns = data.columns.swaplevel(0, 1)
            # Old multi-ticker format: level 0 = ticker already — no change.
        else:
            # Old single-ticker flat format: wrap in a (ticker, field) MultiIndex.
            if len(tickers) == 1:
                data.columns = pd.MultiIndex.from_product(
                    [[tickers[0]], data.columns],
                    names=['ticker', 'field']
                )

        return data

    def _stack_to_multiindex(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape from wide (date × (ticker, field)) to long (ticker, date) × field.

        The stack operation pivots the ticker level (level 0) from columns into
        the row index, producing a MultiIndex (ticker, date).  The subsequent
        swaplevel and sort_index put the index in the canonical (ticker, date)
        order expected by the rest of the pipeline.

        Timezone handling: yfinance returns a tz-aware DatetimeIndex ('UTC' or
        'America/New_York' depending on the version).  The FRED macro data used
        in the aligner has a tz-naive index, so tz information must be removed
        here to avoid alignment errors.

        Pandas compatibility: ``future_stack=True`` (pandas 2.0 behaviour) drops
        rows where all values are NaN, which is the desired behaviour when a
        single ticker is missing some dates.  The try/except handles pandas < 2.0
        where the parameter did not exist yet.

        Args:
            data: DataFrame with (ticker, field) MultiIndex columns.

        Returns:
            DataFrame with MultiIndex (ticker, date) as index.
        """
        try:
            stacked = data.stack(level=0, future_stack=True)
        except TypeError:
            stacked = data.stack(level=0)

        stacked = stacked.swaplevel(0, 1).sort_index()
        stacked.index.names = ['ticker', 'date']

        # Remove timezone — required for join with tz-naive FRED index
        date_level = stacked.index.levels[1]
        if date_level.tz is not None:
            stacked.index = stacked.index.set_levels(
                date_level.tz_convert(None), level='date'
            )

        return stacked
