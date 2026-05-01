"""
FRED Data Loader Module
=======================

Downloads macroeconomic data from the Federal Reserve Economic Data (FRED)
API and prepares it for alignment with the SPY price series.

Role in the pipeline
--------------------
FRED data is used **exclusively for GMM regime detection**, not as a direct
model feature.  Specifically:

- ``vix`` (CBOE Volatility Index, series ``VIXCLS``) is one of the three
  inputs to the Gaussian Mixture Model that classifies market regimes:
  ``(ret_1d, vol_20d_ewm, vix) → regime_state ∈ {Bull, Neutral, Bear}``.
- VIX is a well-established fear gauge: high VIX corresponds to elevated
  realised volatility and bearish market conditions, giving the GMM a
  forward-looking risk signal beyond what OHLCV alone captures.
- VIX is NOT included in the feature matrix **X** used for model training.
  Keeping it out of **X** avoids multicollinearity with the already-included
  ``vol_20d`` and makes the feature set interpretable.

Design decisions that affect data quality
------------------------------------------
1. **Direct CSV API (no API key)** — ``https://fred.stlouisfed.org/graph/
   fredgraph.csv`` is FRED's public export endpoint. Avoids the ``pandas-
   datareader`` library which has had frequent compatibility issues with
   pandas >= 2.0.

2. **'.' → NaN coercion** — FRED uses the literal string '.' for missing
   observations (e.g. VIX before 1990-01-02, bank holidays).  These are
   coerced to NaN via ``pd.to_numeric(..., errors='coerce')`` to prevent
   silent type errors downstream.

3. **Forward fill on non-trading days** — monthly and quarterly series (if
   added) publish a single value per release period.  Forward filling
   propagates the last known value to every subsequent day, which accurately
   reflects what an investor observes: the most recently published figure
   remains the best estimate until the next release.

4. **Release lag** — FRED series published with a delay (e.g. monthly GDP
   released ~30 days after period-end) must be shifted forward by that lag
   to prevent look-ahead bias.  VIX is a real-time daily series (lag = 0).
   The shift is applied here, before alignment, so the aligner receives a
   bias-free FRED DataFrame.

5. **Daily resampling then inner join** — resampling FRED to daily frequency
   and joining on trading dates ensures that every row in the aligned dataset
   has a valid VIX value.  The inner join (handled in DataAligner) drops the
   few dates where FRED has no observation even after forward filling.
"""

from typing import Dict, Any, List, Optional
from io import StringIO
import pandas as pd
import requests
from loguru import logger


class FREDLoader:
    """
    Downloads FRED macroeconomic series and prepares them for alignment
    with yfinance price data.

    Downloads each series individually from the FRED public CSV endpoint
    (no API key required).  Series are forward-filled and resampled to daily
    frequency before being combined into a single DataFrame.

    For each raw series two derived columns are computed:
    - ``{name}_diff``: first difference  — captures direction of change.
    - ``{name}_chg``: percentage change — captures relative magnitude.

    In the current configuration only ``vix`` (VIXCLS) is downloaded.
    The ``vix_diff`` and ``vix_chg`` columns are stored in the HDF5 file
    but the RegimeDetector uses only the ``vix`` level.

    Attributes:
        config (Dict[str, Any]): Full configuration dictionary.
        start_date (str): Download start date (YYYY-MM-DD).
        end_date (Optional[str]): Download end date; None = today.
        add_diff (bool): Whether to add first-difference columns.
        add_pct_change (bool): Whether to add percentage-change columns.
    """

    # Public FRED CSV export endpoint — no authentication required.
    FRED_API_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise FREDLoader from the project configuration.

        Args:
            config: Configuration dictionary.  Reads the 'ingestion' section
                    for start_date, end_date, and fred_transformations.
        """
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.start_date = ingestion_config.get('start_date', '2004-01-01')
        self.end_date = ingestion_config.get('end_date')

        fred_transforms = ingestion_config.get('fred_transformations', {})
        self.add_diff = fred_transforms.get('add_diff', True)
        self.add_pct_change = fred_transforms.get('add_pct_change', True)

        logger.info("FREDLoader initialised")
        logger.info(f"Date range: {self.start_date} to {self.end_date or 'today'}")
        logger.info(
            f"Transformations: diff={self.add_diff}, pct_change={self.add_pct_change}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_series(
        self,
        series_id: str,
        series_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Download a single FRED series from the public CSV endpoint.

        The FRED API returns a two-column CSV (date, value).  Non-numeric
        values — specifically the '.' placeholder that FRED uses for missing
        observations — are coerced to NaN via ``pd.to_numeric``.  This
        handles the VIX gap before 1990-01-02 and any bank-holiday gaps
        without raising a parsing error.

        Args:
            series_id: FRED series identifier (e.g. 'VIXCLS').
            series_name: Column name to assign (e.g. 'vix').
            start_date: Override download start date (YYYY-MM-DD).
            end_date: Override download end date (YYYY-MM-DD).

        Returns:
            Series indexed by date with name ``series_name``.

        Raises:
            ValueError: If the HTTP request fails or returns empty data.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date

        logger.info(f"Downloading FRED series '{series_id}' → column '{series_name}'")

        try:
            params: Dict[str, str] = {'id': series_id, 'cosd': start}
            if end:
                params['coed'] = end

            response = requests.get(self.FRED_API_BASE, params=params, timeout=30)
            response.raise_for_status()

            data = pd.read_csv(
                StringIO(response.text), index_col=0, parse_dates=True
            )

            if data.empty:
                raise ValueError(f"FRED returned empty data for series '{series_id}'")

            series = data.iloc[:, 0]
            series.name = series_name

            # Drop timezone info for consistency with yfinance data
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            # Coerce FRED's '.' missing-value placeholder to NaN
            series = pd.to_numeric(series, errors='coerce')

            n_nan = series.isna().sum()
            logger.success(
                f"'{series_id}': {len(series)} rows  "
                f"({series.index.min().date()} → {series.index.max().date()})  "
                f"NaN: {n_nan}"
            )

            return series

        except Exception as e:
            logger.error(f"Failed to download '{series_id}': {e}")
            raise ValueError(f"Download failed for '{series_id}': {e}") from e

    def download_all_series(self) -> pd.DataFrame:
        """
        Download every FRED series specified in the 'fred_series' config section.

        Release-lag handling
        ~~~~~~~~~~~~~~~~~~~~
        Some FRED series are published with a delay relative to the period they
        describe.  For example, a monthly employment report released on the first
        Friday of the following month has an effective lag of ~30 calendar days.
        Using the raw observation date without shifting would cause the model to
        see future information at training time (look-ahead bias).

        The ``release_lag_days`` config parameter shifts the series forward by
        that many days, so the value published on date *d* is only visible to
        the model from date *d + lag* onwards — matching the real investor's
        information set.

        VIX (VIXCLS) is a real-time daily series: it is published on the same
        day it is observed, so its lag is 0.

        Args:
            (none — reads from self.config)

        Returns:
            DataFrame with one column per series, indexed by date.

        Raises:
            ValueError: If no series are configured or all downloads fail.
        """
        logger.info("Downloading all configured FRED series")

        fred_series_config = self.config.get('fred_series', {})
        if not fred_series_config:
            raise ValueError("No FRED series specified in configuration 'fred_series'")

        logger.info(f"Configured series: {list(fred_series_config.keys())}")

        all_series: Dict[str, pd.Series] = {}
        failed: List[str] = []

        for series_name, series_cfg in fred_series_config.items():
            series_id = series_cfg.get('series_id')
            release_lag = series_cfg.get('release_lag_days', 0)

            try:
                series = self.download_series(series_id, series_name)
                if release_lag > 0:
                    series = series.shift(release_lag)
                    logger.info(
                        f"  '{series_name}': shifted +{release_lag} days "
                        f"(release lag — prevents look-ahead bias)"
                    )
                all_series[series_name] = series

            except Exception as e:
                logger.warning(f"Skipping '{series_name}': {e}")
                failed.append(series_name)

        if not all_series:
            raise ValueError(
                f"Failed to download any FRED series. Failed: {failed}"
            )

        df = pd.DataFrame(all_series)

        logger.info(
            f"Downloaded {len(all_series)}/{len(fred_series_config)} series  "
            f"({len(failed)} failed: {failed or 'none'})"
        )
        logger.info(f"Combined FRED shape: {df.shape}  "
                    f"({df.index.min().date()} → {df.index.max().date()})")

        return df

    def apply_forward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill missing values across all FRED series.

        Non-daily series (monthly, quarterly) have NaN on every day that is
        not a release date.  Forward filling propagates the last published
        value to all subsequent days, which matches what a real investor
        observes: in the absence of a new release, the best available estimate
        is the most recently published one.

        Example (monthly series, released on the 1st):
            2020-01-01: 5.0 (published)
            2020-01-02: NaN → filled to 5.0
            ...
            2020-02-01: 5.2 (new release)

        VIX is a daily series and typically has no gaps except on US market
        holidays; forward filling closes those one-day gaps.

        Args:
            data: DataFrame with FRED series (may contain NaN gaps).

        Returns:
            DataFrame with forward-filled values.
        """
        missing_before = int(data.isnull().sum().sum())
        filled = data.ffill()
        missing_after = int(filled.isnull().sum().sum())

        logger.info(
            f"Forward fill: {missing_before} NaN → {missing_after} NaN "
            f"(filled {missing_before - missing_after} gaps)"
        )

        return filled

    def add_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add first-difference and percentage-change columns for each series.

        These derived features capture the rate of change rather than the
        level of each macroeconomic variable.  Rate-of-change signals are
        often more informative for short-horizon prediction because:
        - The level of VIX may be range-bound in a stable regime, while a
          sudden spike in vix_diff signals a regime transition.
        - Differenced series are closer to stationary, which improves model
          generalisation across different volatility regimes.

        Implementation note: transformations are applied *after* daily
        resampling, so ``_diff`` and ``_chg`` reflect genuine day-over-day
        changes for daily series (VIX) and are non-zero only on release dates
        for lower-frequency series — both are correct by construction.

        Currently only ``vix`` is downloaded, so the output columns are:
        ``vix``, ``vix_diff``, ``vix_chg``.  The RegimeDetector uses only
        ``vix`` (the level); the derived columns are stored in HDF5 but not
        used in the current model.

        Args:
            data: DataFrame with raw FRED series.

        Returns:
            DataFrame with original series plus ``_diff`` and ``_chg`` columns.
        """
        result = data.copy()
        original_cols = data.columns.tolist()

        for col in original_cols:
            if self.add_diff:
                result[f"{col}_diff"] = data[col].diff()
            if self.add_pct_change:
                result[f"{col}_chg"] = data[col].pct_change()

        n_derived = len(result.columns) - len(original_cols)
        logger.success(
            f"Transformations: {len(original_cols)} raw + {n_derived} derived "
            f"= {len(result.columns)} total columns"
        )

        return result

    def prepare_fred_data(self) -> pd.DataFrame:
        """
        Orchestrate the full FRED preparation pipeline.

        Steps
        -----
        1. Download all configured series (with release-lag shifts applied).
        2. Forward-fill missing values (closes gaps for non-daily series).
        3. Resample to daily frequency (fills calendar gaps between releases).
        4. Add ``_diff`` and ``_chg`` derived columns.

        The output is ready for inner-join alignment with yfinance data in
        DataAligner.  After the join, every row in the aligned dataset is
        guaranteed to have a valid ``vix`` value (NaN-free), which is a
        prerequisite for GMM regime detection.

        Returns:
            DataFrame indexed by date with all FRED columns.
        """
        logger.info("=" * 60)
        logger.info("FRED DATA PREPARATION")
        logger.info("=" * 60)

        logger.info("Step 1/4  Download raw series")
        data = self.download_all_series()

        logger.info("Step 2/4  Forward fill")
        data = self.apply_forward_fill(data)

        logger.info("Step 3/4  Resample to daily frequency")
        # resample('D').last() creates an entry for every calendar day;
        # the trailing ffill() fills any remaining gaps (e.g. after the last
        # observed date if end_date is a weekend).
        data = data.resample('D').last().ffill()
        logger.info(f"  After resampling: {data.shape}")

        logger.info("Step 4/4  Add derived transformations")
        data = self.add_transformations(data)

        logger.success(
            f"FRED preparation complete  |  shape: {data.shape}  |  "
            f"columns: {data.columns.tolist()}"
        )
        logger.info("=" * 60)

        return data

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute a quality summary for the prepared FRED data.

        Args:
            data: DataFrame returned by prepare_fred_data().

        Returns:
            Dictionary with series counts, date range, and missing value info.
        """
        raw_cols = [c for c in data.columns if not c.endswith(('_diff', '_chg'))]

        return {
            'num_series': len(raw_cols),
            'total_columns': len(data.columns),
            'num_rows': len(data),
            'date_range': {
                'start': str(data.index.min()),
                'end': str(data.index.max())
            },
            'missing_values': int(data.isnull().sum().sum()),
            'columns': data.columns.tolist()
        }
