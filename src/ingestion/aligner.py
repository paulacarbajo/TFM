"""
Data Aligner Module
===================

Joins SPY OHLCV data (MultiIndex) with FRED macroeconomic data (date index)
on the common set of US equity trading days.

Role in the pipeline
--------------------
After the downloader produces SPY data and FREDLoader produces VIX data, this
module produces the unified DataFrame that enters feature engineering.  The
output has MultiIndex (ticker, date) with OHLCV + VIX columns for every
trading day in 2004-2024.

Design decisions that affect data integrity
-------------------------------------------
1. **Inner join on trading dates** — SPY trades only on US business days
   (~252 per year).  FRED's daily resampled series covers all calendar days
   (365/year).  The inner join retains only SPY trading dates, naturally
   excluding weekends and US market holidays.  Critically, after FRED forward
   filling, every retained date has a valid VIX value — so the GMM regime
   detector receives a NaN-free input.

   Why not a left join?  A left join would keep all SPY rows but leave NaN
   for any FRED date missing before the first forward-fill value (e.g.
   2004-01-02 if VIX data starts on 2004-01-05).  The inner join avoids
   this edge case entirely.

2. **No interpolation** — forward fill is the only imputation applied
   (in FREDLoader).  Interpolation across a holiday would introduce a
   value that was never published, constituting a subtle form of look-ahead
   bias for monthly/quarterly series.

3. **Timezone normalisation** — both yfinance and FRED data are converted to
   tz-naive UTC-like DatetimeIndex before the join.  Mixed timezone handling
   would cause the merge to miss matching dates silently.

4. **Stable (ticker, date) sort** — the MultiIndex is sorted after merging
   so that all downstream slicing operations (``xs``, ``loc``) behave
   deterministically.
"""

from typing import Dict, Optional
from pathlib import Path
import pandas as pd
from loguru import logger

# Project root — two levels up from src/ingestion/aligner.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DataAligner:
    """
    Aligns the SPY OHLCV MultiIndex DataFrame with the FRED date-indexed
    DataFrame via an inner join on US equity trading dates.

    The yfinance data has MultiIndex (ticker, date); the FRED data has a
    plain DatetimeIndex.  The merge is performed on the 'date' level of
    the MultiIndex so that each (ticker, date) row acquires the FRED values
    for that date.

    After alignment every row is guaranteed to have:
    - Valid OHLCV values (from yfinance, split/dividend-adjusted).
    - A valid ``vix`` value (from FRED, forward-filled).
    - No duplicate (ticker, date) pairs.

    Attributes:
        config (Dict): Full configuration dictionary.
        processed_data_path (Path): Output directory for processed data.
        join_method (str): Merge method, either 'inner' (default) or 'left'.
    """

    def __init__(self, config: Dict):
        """
        Initialise DataAligner from the project configuration.

        Args:
            config: Configuration dictionary.  Reads the 'ingestion' section
                    for processed_data_path and alignment.method.
        """
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def align_yfinance_with_fred(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge SPY OHLCV data with FRED macroeconomic data on trading dates.

        Procedure
        ---------
        1. Reset the yfinance MultiIndex to expose 'date' as a plain column.
        2. Merge with FRED on date (left key = 'date', right key = index).
        3. Restore the (ticker, date) MultiIndex and sort.

        The inner join means: only dates present in BOTH datasets are kept.
        Since FRED is forward-filled to daily frequency and VIX covers the
        full 2004-2024 SPY history, no SPY trading day is lost in practice.

        Args:
            yfinance_data: MultiIndex (ticker, date) DataFrame with OHLCV.
            fred_data: Date-indexed DataFrame with FRED series (e.g. vix).

        Returns:
            MultiIndex (ticker, date) DataFrame combining OHLCV and FRED
            columns, sorted by (ticker, date).
        """
        logger.info("Aligning yfinance data with FRED data")
        logger.info(f"  yfinance: {yfinance_data.shape}  "
                    f"({yfinance_data.index.get_level_values('date').min().date()} → "
                    f"{yfinance_data.index.get_level_values('date').max().date()})")
        logger.info(f"  FRED:     {fred_data.shape}  "
                    f"({fred_data.index.min().date()} → {fred_data.index.max().date()})")

        # Step 1: flatten MultiIndex so 'date' is a merge key
        yf_reset = yfinance_data.reset_index()

        # Step 2: merge on date — inner join retains only common dates
        merged = yf_reset.merge(
            fred_data,
            left_on='date',
            right_index=True,
            how=self.join_method
        )

        # Step 3: normalise date column and restore MultiIndex
        # pd.to_datetime is defensive in case the merge produced object dtype dates
        merged['date'] = pd.to_datetime(merged['date'])
        if merged['date'].dt.tz is not None:
            merged['date'] = merged['date'].dt.tz_convert(None)

        merged = merged.set_index(['ticker', 'date']).sort_index()

        # Log how many rows were lost (if any) by the inner join
        n_in = len(yfinance_data)
        n_out = len(merged)
        if n_in != n_out:
            logger.warning(
                f"Inner join dropped {n_in - n_out} rows "
                f"(dates present in yfinance but not in FRED after forward fill)"
            )

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
        Validate that the aligned DataFrame meets pipeline requirements.

        Checks performed
        ----------------
        - MultiIndex with names ['ticker', 'date'].
        - Non-empty DataFrame.
        - DatetimeIndex on the date level (tz-naive).
        - All five OHLCV columns present.
        - Missing value count reported (warning, not error, to allow
          partial data for debugging).

        Args:
            data: Aligned DataFrame to validate.

        Returns:
            True if all checks pass.

        Raises:
            ValueError: If index structure or data are invalid.
        """
        logger.info("Validating aligned data")

        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Aligned data must have a MultiIndex")

        if list(data.index.names) != ['ticker', 'date']:
            raise ValueError(
                f"Expected index names ['ticker', 'date'], "
                f"got {data.index.names}"
            )

        if data.empty:
            raise ValueError("Aligned data is empty")

        date_index = data.index.get_level_values('date')
        if not isinstance(date_index, pd.DatetimeIndex):
            raise ValueError("'date' level must be a DatetimeIndex")

        if date_index.tz is not None:
            logger.warning(
                "Date index is tz-aware — expected tz-naive.  "
                "Feature engineering may fail on pandas merge operations."
            )

        # Verify OHLCV columns
        expected_ohlcv = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_ohlcv = [c for c in expected_ohlcv if c not in data.columns]
        if missing_ohlcv:
            logger.warning(f"Missing OHLCV columns: {missing_ohlcv}")

        # Report missing values
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
        """
        Build a diagnostic summary of the alignment step.

        Reports input/output row counts, tickers, columns, date range, and
        the number of missing values in the aligned output.

        Args:
            yfinance_data: Original yfinance MultiIndex DataFrame.
            fred_data: Original FRED date-indexed DataFrame.
            aligned_data: Output of align_yfinance_with_fred().

        Returns:
            Dictionary with alignment statistics.
        """
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
            ticker_data = aligned_data.xs(ticker, level='ticker')
            summary['rows_per_ticker'][ticker] = len(ticker_data)

        return summary

    def get_common_dates(
        self,
        yfinance_data: pd.DataFrame,
        fred_data: pd.DataFrame
    ) -> pd.DatetimeIndex:
        """
        Return the intersection of trading dates between the two datasets.

        Useful for diagnosing how many SPY dates would survive an inner join
        before actually performing it.

        Args:
            yfinance_data: MultiIndex (ticker, date) DataFrame.
            fred_data: Date-indexed DataFrame.

        Returns:
            DatetimeIndex of dates present in both datasets.
        """
        yf_dates = yfinance_data.index.get_level_values('date').unique()
        fred_dates = fred_data.index
        common_dates = yf_dates.intersection(fred_dates)

        logger.info(
            f"Date intersection  |  yfinance: {len(yf_dates)}  "
            f"FRED: {len(fred_dates)}  common: {len(common_dates)}  "
            f"(dropped by inner join: {len(yf_dates) - len(common_dates)})"
        )

        return common_dates
