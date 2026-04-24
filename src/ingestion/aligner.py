"""
Data Aligner Module

Aligns yfinance data (MultiIndex) with FRED economic data via an inner join
on trading dates. The resulting DataFrame has MultiIndex (ticker, date) with
both OHLCV and macroeconomic columns.
"""

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
from loguru import logger

# Project root — two levels up from src/ingestion/aligner.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DataAligner:
    """
    Aligns yfinance data with FRED economic data.

    The yfinance data has MultiIndex (ticker, date) while FRED data has
    a simple date index. Alignment is performed via an inner join on the
    date level, so only dates present in both datasets are retained.

    Because each ticker has its own set of trading dates (USO starts in 2006,
    SPY in 2004), the inner join is performed per row of the MultiIndex —
    each (ticker, date) pair finds its corresponding FRED row independently.
    This means SPY retains its full history from 2004 while USO starts in 2006,
    without artificial NaNs.

    Attributes:
        config (Dict): Configuration dictionary
        processed_data_path (Path): Path to store processed data
        join_method (str): Join method for alignment (default 'inner')
    """

    def __init__(self, config: Dict):
        """
        Initialize the DataAligner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.processed_data_path = PROJECT_ROOT / ingestion_config.get(
            'processed_data_path', 'data/processed/'
        )
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        alignment_config = ingestion_config.get('alignment', {})
        self.join_method = alignment_config.get('method', 'inner')

        logger.info(f"DataAligner initialized with path: {self.processed_data_path}")
        logger.info(f"Join method: {self.join_method}")

    def align_yfinance_with_fred(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align yfinance MultiIndex data with FRED date-indexed data.

        Merges on the date level so that each (ticker, date) row receives
        the corresponding FRED values for that date.

        Args:
            yfinance_data: DataFrame with MultiIndex (ticker, date) and OHLCV columns
            fred_data: DataFrame with date index and macro columns

        Returns:
            DataFrame with MultiIndex (ticker, date) containing both
            OHLCV and macroeconomic columns
        """
        logger.info("Aligning yfinance data with FRED data")
        logger.info(f"yfinance shape: {yfinance_data.shape}")
        logger.info(f"FRED shape: {fred_data.shape}")

        yf_reset = yfinance_data.reset_index()

        merged = yf_reset.merge(
            fred_data,
            left_on='date',
            right_index=True,
            how=self.join_method
        )

        merged['date'] = pd.to_datetime(merged['date'])
        if merged['date'].dt.tz is not None:
            merged['date'] = merged['date'].dt.tz_convert(None)

        merged = merged.set_index(['ticker', 'date'])
        merged = merged.sort_index()

        logger.success(f"Aligned data shape: {merged.shape}")
        logger.info(
            f"Date range: {merged.index.get_level_values('date').min()} to "
            f"{merged.index.get_level_values('date').max()}"
        )
        logger.info(
            f"Tickers: "
            f"{merged.index.get_level_values('ticker').unique().tolist()}"
        )
        logger.info(f"Total columns: {len(merged.columns)}")

        return merged

    def get_alignment_summary(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame,
        aligned_data: pd.DataFrame
    ) -> Dict:
        """
        Get summary of the alignment process.

        Args:
            yfinance_data: Original yfinance data
            fred_data: Original FRED data
            aligned_data: Aligned DataFrame

        Returns:
            Dictionary with alignment summary
        """
        tickers = aligned_data.index.get_level_values('ticker').unique().tolist()

        summary = {
            'input_yfinance_rows': len(yfinance_data),
            'input_fred_rows': len(fred_data),
            'output_rows': len(aligned_data),
            'num_tickers': len(tickers),
            'tickers': tickers,
            'yfinance_columns': yfinance_data.columns.tolist(),
            'fred_columns': fred_data.columns.tolist(),
            'total_columns': len(aligned_data.columns),
            'date_range': {
                'start': str(aligned_data.index.get_level_values('date').min()),
                'end': str(aligned_data.index.get_level_values('date').max())
            },
            'missing_values': int(aligned_data.isnull().sum().sum()),
            'rows_per_ticker': {}
        }

        for ticker in tickers:
            ticker_data = aligned_data.xs(ticker, level='ticker')
            summary['rows_per_ticker'][ticker] = len(ticker_data)

        return summary

    def validate_alignment(self, data: pd.DataFrame) -> bool:
        """
        Validate the aligned data structure.

        Args:
            data: Aligned DataFrame to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating aligned data")

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex")

        if data.index.names != ['ticker', 'date']:
            raise ValueError(
                f"Index names must be ['ticker', 'date'], "
                f"got {data.index.names}"
            )

        if data.empty:
            raise ValueError("Aligned data is empty")

        date_index = data.index.get_level_values('date')
        if not isinstance(date_index, pd.DatetimeIndex):
            raise ValueError("Date level must be DatetimeIndex")

        if date_index.tz is not None:
            logger.warning("Date index has timezone — should be timezone-naive")

        expected_ohlcv = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_ohlcv = [col for col in expected_ohlcv if col not in data.columns]
        if missing_ohlcv:
            logger.warning(f"Missing OHLCV columns: {missing_ohlcv}")

        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Data contains {missing_count} missing values")
            missing_by_col = data.isnull().sum()
            missing_by_col = missing_by_col[missing_by_col > 0]
            logger.debug(f"Missing values by column:\n{missing_by_col}")

        logger.success("Validation passed")
        return True

    def get_common_dates(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame
    ) -> pd.DatetimeIndex:
        """
        Get the intersection of trading dates between yfinance and FRED data.

        Useful for diagnosing how many dates are lost by the inner join.

        Args:
            yfinance_data: DataFrame with MultiIndex (ticker, date)
            fred_data: DataFrame with date index

        Returns:
            DatetimeIndex of common dates
        """
        yf_dates = yfinance_data.index.get_level_values('date').unique()
        fred_dates = fred_data.index
        common_dates = yf_dates.intersection(fred_dates)

        logger.info(f"yfinance dates: {len(yf_dates)}")
        logger.info(f"FRED dates: {len(fred_dates)}")
        logger.info(f"Common dates: {len(common_dates)}")

        return common_dates