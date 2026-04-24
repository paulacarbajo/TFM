"""
FRED Data Loader Module

Downloads and transforms economic data from FRED (Federal Reserve Economic Data).
Uses direct API calls to the FRED CSV endpoint for compatibility with modern
pandas versions.
"""

from typing import Dict, Any, Optional
from io import StringIO
import pandas as pd
import requests
from loguru import logger


class FREDLoader:
    """
    Downloads economic data from FRED and applies transformations.

    Downloads each series individually using the FRED public CSV API
    (no API key required). Series with different frequencies (daily, monthly,
    quarterly) are forward-filled and resampled to a common daily frequency
    before being combined into a single DataFrame.

    For each series, two derived columns are added:
    - {series_name}_diff: first difference
    - {series_name}_chg: percentage change

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        start_date (str): Start date for data download
        end_date (Optional[str]): End date for data download
        add_diff (bool): Whether to add first-difference columns
        add_pct_change (bool): Whether to add percentage-change columns
    """

    FRED_API_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FREDLoader.

        Args:
            config: Configuration dictionary with FRED series specifications
        """
        self.config = config
        ingestion_config = config.get('ingestion', {})

        self.start_date = ingestion_config.get('start_date', '2004-01-01')
        self.end_date = ingestion_config.get('end_date')

        fred_transforms = ingestion_config.get('fred_transformations', {})
        self.add_diff = fred_transforms.get('add_diff', True)
        self.add_pct_change = fred_transforms.get('add_pct_change', True)

        logger.info("FREDLoader initialized")
        logger.info(f"Date range: {self.start_date} to {self.end_date or 'today'}")
        logger.info(
            f"Transformations: diff={self.add_diff}, "
            f"pct_change={self.add_pct_change}"
        )

    def download_series(
        self,
        series_id: str,
        series_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Download a single FRED series using the public CSV endpoint.

        FRED occasionally returns '.' instead of a numeric value when
        a data point is not available. These are coerced to NaN via
        pd.to_numeric(..., errors='coerce').

        Args:
            series_id: FRED series ID (e.g., 'VIXCLS')
            series_name: Column name to assign to the series (e.g., 'vix')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            Series with the downloaded data, indexed by date

        Raises:
            ValueError: If download fails or returns empty data
        """
        start = start_date or self.start_date
        end = end_date or self.end_date

        logger.info(f"Downloading FRED series {series_id} ({series_name})")

        try:
            params = {'id': series_id, 'cosd': start}
            if end:
                params['coed'] = end

            response = requests.get(self.FRED_API_BASE, params=params)
            response.raise_for_status()

            data = pd.read_csv(
                StringIO(response.text), index_col=0, parse_dates=True
            )

            if data.empty:
                raise ValueError(f"No data returned for {series_id}")

            series = data.iloc[:, 0]
            series.name = series_name

            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            # Coerce non-numeric values (e.g. '.') to NaN
            series = pd.to_numeric(series, errors='coerce')

            logger.success(
                f"Downloaded {series_id}: {len(series)} rows, "
                f"range {series.index.min()} to {series.index.max()}"
            )

            return series

        except Exception as e:
            logger.error(f"Failed to download {series_id}: {str(e)}")
            raise ValueError(f"Download failed for {series_id}: {str(e)}")

    def download_all_series(self) -> pd.DataFrame:
        """
        Download all FRED series specified in configuration.

        Series that fail to download are skipped with a warning.
        At least one series must succeed.

        Returns:
            DataFrame with all FRED series (index is date, columns are series names)

        Raises:
            ValueError: If all series fail to download
        """
        logger.info("Starting download of all FRED series")

        fred_series = self.config.get('fred_series', {})

        if not fred_series:
            raise ValueError("No FRED series specified in configuration")

        logger.info(f"Configured series: {list(fred_series.keys())}")

        all_series = {}
        failed_series = []

        for series_name, series_config in fred_series.items():
            series_id = series_config.get('series_id')
            try:
                series = self.download_series(series_id, series_name)
                all_series[series_name] = series
            except Exception as e:
                logger.warning(f"Skipping {series_name}: {str(e)}")
                failed_series.append(series_name)

        if not all_series:
            raise ValueError("Failed to download any FRED series")

        df = pd.DataFrame(all_series)

        logger.info(
            f"Downloaded {len(all_series)}/{len(fred_series)} series successfully"
        )
        if failed_series:
            logger.warning(f"Failed series: {failed_series}")

        logger.info(f"Combined FRED data shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        return df

    def apply_forward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply forward fill to handle missing values in FRED data.

        Monthly and quarterly series have NaN on days without a new release.
        Forward filling propagates the last known value, consistent with what
        a real investor would observe on those days.

        Args:
            data: DataFrame with FRED series

        Returns:
            DataFrame with forward-filled values
        """
        logger.info("Applying forward fill to FRED data")

        missing_before = data.isnull().sum().sum()
        filled = data.ffill()
        missing_after = filled.isnull().sum().sum()

        logger.info(f"Missing values: {missing_before} -> {missing_after}")

        return filled

    def add_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add first-difference and percentage-change columns for each series.

        These derived features capture the direction and speed of change
        in each macroeconomic variable, which are often more stationary
        and informative for prediction than the raw level.

        Args:
            data: DataFrame with FRED series

        Returns:
            DataFrame with original series plus _diff and _chg columns
        """
        logger.info("Adding transformations to FRED data")

        result = data.copy()
        original_cols = data.columns.tolist()

        for col in original_cols:
            if self.add_diff:
                result[f"{col}_diff"] = data[col].diff()
                logger.debug(f"Added {col}_diff")
            if self.add_pct_change:
                result[f"{col}_chg"] = data[col].pct_change()
                logger.debug(f"Added {col}_chg")

        n_new = len(result.columns) - len(original_cols)
        logger.success(
            f"Transformations added: {len(original_cols)} original + "
            f"{n_new} derived = {len(result.columns)} total columns"
        )

        return result

    def prepare_fred_data(self) -> pd.DataFrame:
        """
        Download and prepare all FRED data with transformations.

        Pipeline:
        1. Download all series
        2. Forward fill missing values
        3. Resample to daily frequency (fills calendar gaps)
        4. Add _diff and _chg transformations

        Returns:
            DataFrame with all FRED data ready for alignment with yfinance
        """
        logger.info("=" * 60)
        logger.info("FRED DATA PREPARATION PIPELINE")
        logger.info("=" * 60)

        logger.info("Step 1: Downloading FRED series")
        data = self.download_all_series()

        logger.info("Step 2: Forward filling missing values")
        data = self.apply_forward_fill(data)

        logger.info("Step 3: Resampling to daily frequency")
        data = data.resample('D').last().ffill()
        logger.info(f"After resampling: {data.shape}")

        logger.info("Step 4: Adding transformations")
        data = self.add_transformations(data)

        logger.info("=" * 60)
        logger.success("FRED data preparation complete")
        logger.info(f"Final shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")
        logger.info("=" * 60)

        return data

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for FRED data.

        Args:
            data: DataFrame with FRED data

        Returns:
            Dictionary with summary statistics
        """
        return {
            'num_series': len([
                col for col in data.columns
                if not col.endswith(('_diff', '_chg'))
            ]),
            'total_columns': len(data.columns),
            'num_rows': len(data),
            'date_range': {
                'start': str(data.index.min()),
                'end': str(data.index.max())
            },
            'missing_values': int(data.isnull().sum().sum()),
            'columns': data.columns.tolist()
        }