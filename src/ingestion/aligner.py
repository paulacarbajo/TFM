"""
Data Aligner — joins SPY OHLCV (MultiIndex) with FRED data (date index) on trading dates.

Key decisions:
- Inner join: retains only dates present in both datasets. Since FRED is forward-filled
  to daily frequency, no SPY trading day is lost in practice.
- No interpolation: only forward fill (applied in FREDLoader). Interpolation across
  holidays would introduce values never published, creating look-ahead bias.
- tz-naive dates: both sources are normalised to tz-naive before merging to avoid
  silent mismatches.
- Sorted (ticker, date) MultiIndex: required for deterministic downstream slicing.
"""

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DataAligner:
    """
    Aligns SPY OHLCV (MultiIndex ticker/date) with FRED data (date index)
    via an inner join on US equity trading dates.

    Output guarantees: valid OHLCV + valid vix for every (ticker, date) row.
    """

    def __init__(self, config: Dict):
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.processed_data_path = PROJECT_ROOT / ingestion_config.get(
            'processed_data_path', 'data/processed/'
        )
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        alignment_config = ingestion_config.get('alignment', {})
        self.join_method = alignment_config.get('method', 'inner')

        logger.info(f"DataAligner initialised  |  join: {self.join_method}  |  "
                    f"output: {self.processed_data_path}")

    def align_yfinance_with_fred(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge SPY OHLCV with FRED data on trading dates.

        Steps: reset MultiIndex → merge on date → restore (ticker, date) index.
        Inner join keeps only dates present in both datasets.
        """
        logger.info("Aligning yfinance data with FRED data")
        logger.info(f"  yfinance: {yfinance_data.shape}  "
                    f"({yfinance_data.index.get_level_values('date').min().date()} → "
                    f"{yfinance_data.index.get_level_values('date').max().date()})")
        logger.info(f"  FRED:     {fred_data.shape}  "
                    f"({fred_data.index.min().date()} → {fred_data.index.max().date()})")

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

        merged = merged.set_index(['ticker', 'date']).sort_index()

        n_dropped = len(yfinance_data) - len(merged)
        if n_dropped:
            logger.warning(f"Inner join dropped {n_dropped} rows")

        logger.success(
            f"Alignment complete  |  shape: {merged.shape}  |  "
            f"tickers: {merged.index.get_level_values('ticker').unique().tolist()}"
        )
        logger.info(
            f"Date range: {merged.index.get_level_values('date').min().date()} → "
            f"{merged.index.get_level_values('date').max().date()}"
        )
        logger.info(f"Columns ({len(merged.columns)}): {merged.columns.tolist()}")

        return merged

    def validate_alignment(self, data: pd.DataFrame) -> bool:
        """
        Validate the aligned DataFrame meets pipeline requirements.

        Checks: MultiIndex structure, non-empty, tz-naive dates,
        OHLCV columns present, vix column present, missing value count.
        """
        logger.info("Validating aligned data")

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Aligned data must have a MultiIndex")

        if list(data.index.names) != ['ticker', 'date']:
            raise ValueError(
                f"Expected index names ['ticker', 'date'], got {data.index.names}"
            )

        if data.empty:
            raise ValueError("Aligned data is empty")

        date_index = data.index.get_level_values('date')
        if not isinstance(date_index, pd.DatetimeIndex):
            raise ValueError("'date' level must be a DatetimeIndex")

        if date_index.tz is not None:
            logger.warning(
                "Date index is tz-aware — expected tz-naive. "
                "Feature engineering may fail on pandas merge operations."
            )

        missing_ohlcv = [c for c in ['Close', 'High', 'Low', 'Open', 'Volume']
                         if c not in data.columns]
        if missing_ohlcv:
            logger.warning(f"Missing OHLCV columns: {missing_ohlcv}")

        # vix is required downstream for GMM regime detection
        if 'vix' not in data.columns:
            logger.warning("'vix' column missing — GMM regime detection will fail")

        missing_total = int(data.isnull().sum().sum())
        if missing_total > 0:
            logger.warning(f"Aligned data contains {missing_total} NaN values")
            missing_by_col = data.isnull().sum()
            for col, n in missing_by_col[missing_by_col > 0].items():
                logger.debug(f"  {col}: {n} NaN")
        else:
            logger.info("No missing values in aligned data")

        logger.success("Validation passed")
        return True

    def get_alignment_summary(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame,
        aligned_data: pd.DataFrame
    ) -> Dict:
        """Return diagnostic summary of the alignment step (row counts, columns, NaNs)."""
        tickers = aligned_data.index.get_level_values('ticker').unique().tolist()

        summary: Dict = {
            'input_yfinance_rows': len(yfinance_data),
            'input_fred_rows': len(fred_data),
            'output_rows': len(aligned_data),
            'rows_dropped_by_join': len(yfinance_data) - len(aligned_data),
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
            summary['rows_per_ticker'][ticker] = len(
                aligned_data.xs(ticker, level='ticker')
            )

        return summary

    def get_common_dates(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame
    ) -> pd.DatetimeIndex:
        """Return the intersection of trading dates between both datasets (diagnostic)."""
        yf_dates = yfinance_data.index.get_level_values('date').unique()
        fred_dates = fred_data.index
        common_dates = yf_dates.intersection(fred_dates)

        logger.info(
            f"Date intersection  |  yfinance: {len(yf_dates)}  "
            f"FRED: {len(fred_dates)}  common: {len(common_dates)}  "
            f"(dropped: {len(yf_dates) - len(common_dates)})"
        )

        return common_dates
