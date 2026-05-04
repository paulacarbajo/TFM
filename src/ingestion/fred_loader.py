"""
FRED Data Loader — downloads macroeconomic series from the Federal Reserve API.

Role in pipeline:
- VIX (VIXCLS) is the only series currently downloaded.
- VIX is used exclusively as a GMM input for regime detection (not as a model feature).
- vix_diff and vix_chg are computed but not used by any model — kept for potential future use.

Key decisions:
- Direct CSV endpoint (no API key): avoids pandas-datareader compatibility issues.
- '.' → NaN coercion: FRED uses '.' for missing values; pd.to_numeric handles this.
- Forward fill: propagates last known value to non-release days (correct for investor info set).
- release_lag_days: shifts series forward to prevent look-ahead bias on delayed publications.
  Note: shift() moves by rows (trading days for daily series), not calendar days.
  VIX has lag=0 so this has no current effect.
"""

from typing import Dict, Any, List, Optional
from io import StringIO
import pandas as pd
import requests
from loguru import logger


class FREDLoader:
    """
    Downloads FRED series and prepares them for alignment with yfinance data.

    Each series is forward-filled and resampled to daily frequency.
    Optional _diff and _chg columns are added per series (vix_diff, vix_chg).
    """

    FRED_API_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.start_date = ingestion_config.get('start_date', '2004-01-01')
        self.end_date = ingestion_config.get('end_date')

        fred_transforms = ingestion_config.get('fred_transformations', {})
        self.add_diff = fred_transforms.get('add_diff', True)
        self.add_pct_change = fred_transforms.get('add_pct_change', True)

        logger.info("FREDLoader initialised")
        logger.info(f"Date range: {self.start_date} to {self.end_date or 'today'}")

    def download_series(
        self,
        series_id: str,
        series_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Download a single FRED series from the public CSV endpoint.
        FRED's '.' placeholder for missing values is coerced to NaN.
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

            data = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)

            if data.empty:
                raise ValueError(f"FRED returned empty data for series '{series_id}'")

            series = data.iloc[:, 0]
            series.name = series_name

            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)

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
        Download every series in config['fred_series'], applying release_lag_days shift.
        Shift is row-based (suitable for daily series; would need adjustment for monthly).
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
                    logger.info(f"  '{series_name}': shifted +{release_lag} rows (release lag)")
                all_series[series_name] = series

            except Exception as e:
                logger.warning(f"Skipping '{series_name}': {e}")
                failed.append(series_name)

        if not all_series:
            raise ValueError(f"Failed to download any FRED series. Failed: {failed}")

        df = pd.DataFrame(all_series)

        logger.info(
            f"Downloaded {len(all_series)}/{len(fred_series_config)} series  "
            f"({len(failed)} failed: {failed or 'none'})"
        )
        logger.info(
            f"Combined FRED shape: {df.shape}  "
            f"({df.index.min().date()} → {df.index.max().date()})"
        )

        return df

    def apply_forward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill NaN gaps (holiday closures, non-release days)."""
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
        Add _diff (first difference) and _chg (pct change) columns per series.
        Currently computed for vix but not consumed by any model — stored in HDF5 only.
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
        Full FRED preparation pipeline:
        1. Download all configured series (with release-lag shifts).
        2. Forward-fill missing values.
        3. Resample to daily frequency.
        4. Add _diff and _chg derived columns.

        Output is ready for inner-join alignment with yfinance data in DataAligner.
        """
        logger.info("=" * 60)
        logger.info("FRED DATA PREPARATION")
        logger.info("=" * 60)

        logger.info("Step 1/4  Download raw series")
        data = self.download_all_series()

        logger.info("Step 2/4  Forward fill")
        data = self.apply_forward_fill(data)

        logger.info("Step 3/4  Resample to daily frequency")
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

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return quality summary dict (series counts, date range, missing values)."""
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
