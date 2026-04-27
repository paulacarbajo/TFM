"""
Data Downloader Module

Downloads financial data from yfinance and creates MultiIndex DataFrame.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import yfinance as yf
from loguru import logger

# Project root — two levels up from src/ingestion/downloader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set yfinance cache location relative to project root
yf.set_tz_cache_location(str(PROJECT_ROOT / "data/cache"))


class DataDownloader:
    """
    Downloads financial market data using yfinance and creates MultiIndex structure.

    The resulting DataFrame has MultiIndex (ticker, date) for compatibility with
    the feature engineering pipeline.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        raw_data_path (Path): Path to store raw data files
        start_date (str): Start date for data download
        end_date (Optional[str]): End date for data download
        frequency (str): Data frequency (default '1d')
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataDownloader.

        Args:
            config: Configuration dictionary with data source specifications
        """
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.raw_data_path = PROJECT_ROOT / ingestion_config.get('raw_data_path', 'data/raw/')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.start_date = ingestion_config.get('start_date', '2004-01-01')
        self.end_date = ingestion_config.get('end_date')
        self.frequency = ingestion_config.get('frequency', '1d')

        logger.info(f"DataDownloader initialized with path: {self.raw_data_path}")
        logger.info(f"Date range: {self.start_date} to {self.end_date or 'today'}")

    def download_multiple_tickers(
        self,
        tickers: list,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download historical data for multiple tickers and create MultiIndex DataFrame.

        Args:
            tickers: List of ticker symbols (e.g., ['SPY', 'USO'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with MultiIndex (ticker, date) and OHLCV columns

        Raises:
            ValueError: If download fails for all tickers
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

            # Check if data is empty and try fallback for single ticker
            if data.empty and len(tickers) == 1:
                logger.warning(f"yf.download() returned empty data for {tickers[0]}, trying fallback...")
                ticker = tickers[0]
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(
                    start=start,
                    end=end,
                    interval=self.frequency,
                    auto_adjust=True
                )
                
                # Debug: log what yfinance returned
                logger.debug(f"Fallback data columns: {data.columns.tolist()}")
                logger.debug(f"Fallback data head:\n{data.head(2)}")
                
                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Rename columns to standard OHLCV names if needed
                data.columns = [str(col).strip() for col in data.columns]
                
                # Drop timezone from index if present
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                if data.empty:
                    raise ValueError(f"No data returned for ticker: {ticker} (both methods failed)")
                
                logger.success(f"Fallback successful for {ticker}")

            # Final check after potential fallback
            if data.empty:
                raise ValueError(f"No data returned for tickers: {tickers}")

            # yfinance returns simple columns for a single ticker
            # and MultiIndex columns for multiple tickers — normalise to MultiIndex
            if len(tickers) == 1:
                ticker = tickers[0]
                data.columns = pd.MultiIndex.from_product(
                    [[ticker], data.columns],
                    names=['ticker', 'field']
                )

            # Reshape from (date, (ticker, field)) to (ticker, date) x field
            stacked = data.stack(level=0)
            stacked = stacked.swaplevel(0, 1)
            stacked = stacked.sort_index()
            stacked.index.names = ['ticker', 'date']

            # Remove timezone info if present — required for join with FRED
            date_level = stacked.index.levels[1]
            if date_level.tz is not None:
                stacked.index = stacked.index.set_levels(
                    date_level.tz_convert(None),
                    level='date'
                )

            expected_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            available_cols = [col for col in expected_cols if col in stacked.columns]
            stacked = stacked[available_cols]

            logger.success(
                f"Downloaded data with shape {stacked.shape}, "
                f"tickers: {stacked.index.get_level_values('ticker').unique().tolist()}"
            )
            logger.info(
                f"Date range: {stacked.index.get_level_values('date').min()} to "
                f"{stacked.index.get_level_values('date').max()}"
            )

            return stacked

        except Exception as e:
            logger.error(f"Failed to download tickers {tickers}: {str(e)}")
            raise ValueError(f"Download failed for {tickers}: {str(e)}")

    def download_all_assets(self) -> pd.DataFrame:
        """
        Download all assets specified in configuration.

        Returns:
            DataFrame with MultiIndex (ticker, date) containing all assets
        """
        logger.info("Starting download of all configured assets")

        data_sources = self.config.get('data_sources', {})
        tickers = [asset_config.get('ticker') for asset_config in data_sources.values()]

        if not tickers:
            raise ValueError("No tickers specified in configuration")

        logger.info(f"Configured tickers: {tickers}")

        data = self.download_multiple_tickers(tickers)

        for ticker in data.index.get_level_values('ticker').unique():
            ticker_data = data.xs(ticker, level='ticker')
            logger.info(
                f"{ticker}: {len(ticker_data)} rows, "
                f"date range {ticker_data.index.min()} to {ticker_data.index.max()}"
            )

        return data

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for downloaded data.

        Args:
            data: DataFrame with MultiIndex (ticker, date)

        Returns:
            Dictionary with summary statistics
        """
        tickers = data.index.get_level_values('ticker').unique().tolist()

        summary = {
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
