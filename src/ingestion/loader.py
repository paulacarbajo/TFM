"""
Data Loader — persists and retrieves pipeline DataFrames using HDF5.

Two keys in a single file (data/processed/assets.h5):
- 'data_raw':             aligned OHLCV + VIX, before feature engineering.
- 'engineered_features':  11 technical features + labels + forward returns.

All training scripts read from 'engineered_features' via load_engineered_features().

Format: HDF5 table (supports MultiIndex round-trip), blosc compression level 9.
data_columns=True enables column-level queries without loading the full file.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_COMPLEVEL = 9
_DEFAULT_COMPLIB = 'blosc'


class DataLoader:
    """
    Saves and loads pipeline DataFrames from a single HDF5 file.

    Keys:
    - 'data_raw':            OHLCV + VIX (pre-feature engineering).
    - 'engineered_features': features + labels + forward returns.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ingestion_config = config.get('ingestion', {})

        hdf5_file = ingestion_config.get('hdf5_file', 'data/processed/assets.h5')
        self.hdf5_path = PROJECT_ROOT / hdf5_file
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        hdf5_keys = ingestion_config.get('hdf5_keys', {})
        self.raw_data_key = hdf5_keys.get('raw_data', 'data_raw')
        self.features_key = hdf5_keys.get('engineered_features', 'engineered_features')

        logger.info(f"DataLoader initialised  |  file: {self.hdf5_path}")
        logger.info(f"Keys: raw='{self.raw_data_key}', features='{self.features_key}'")

    # ------------------------------------------------------------------
    # Core persistence
    # ------------------------------------------------------------------

    def save_to_hdf5(
        self,
        data: pd.DataFrame,
        key: Optional[str] = None,
        mode: str = 'a',
        complevel: int = _DEFAULT_COMPLEVEL,
        complib: str = _DEFAULT_COMPLIB
    ) -> None:
        """
        Persist a DataFrame to the HDF5 file.

        mode='a' appends/updates a key without rewriting the whole file.
        mode='w' overwrites the entire file (used for the first write in main.py).
        """
        key = key or self.raw_data_key

        logger.info(f"Saving to HDF5  |  key: '{key}'  |  shape: {data.shape}")

        try:
            with pd.HDFStore(
                self.hdf5_path, mode=mode, complevel=complevel, complib=complib
            ) as store:
                store.put(key, data, format='table', data_columns=True)

            file_size_mb = self.hdf5_path.stat().st_size / (1024 * 1024)
            logger.success(f"Saved '{key}'  |  file size: {file_size_mb:.2f} MB")

        except Exception as e:
            logger.error(f"HDF5 save failed for key '{key}': {e}")
            raise ValueError(f"HDF5 save failed: {e}") from e

    def load_from_hdf5(self, key: Optional[str] = None) -> pd.DataFrame:
        """
        Load a DataFrame from the HDF5 file.
        MultiIndex and column dtypes are restored exactly as saved.

        Raises FileNotFoundError if the file does not exist.
        Raises ValueError if the key is absent.
        """
        key = key or self.raw_data_key

        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {self.hdf5_path}\n"
                "Run 'python main.py' to generate it."
            )

        logger.info(f"Loading from HDF5  |  key: '{key}'")

        try:
            with pd.HDFStore(self.hdf5_path, mode='r') as store:
                available = list(store.keys())
                # store.keys() returns keys with leading '/' (e.g. '/data_raw');
                # normalise before comparison, store.get() handles both forms.
                norm_key = key if key.startswith('/') else f'/{key}'
                if norm_key not in available:
                    raise ValueError(
                        f"Key '{key}' not found. Available keys: {available}"
                    )
                data = store.get(key)

            logger.success(f"Loaded '{key}'  |  shape: {data.shape}")
            return data

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"HDF5 load failed for key '{key}': {e}")
            raise ValueError(f"HDF5 load failed: {e}") from e

    # ------------------------------------------------------------------
    # Named accessors (primary interface for training scripts)
    # ------------------------------------------------------------------

    def load_engineered_features(self) -> pd.DataFrame:
        """
        Load the fully processed dataset. Entry point for all training scripts.

        Contents:
        - OHLCV: Close, High, Low, Open, Volume (split/dividend-adjusted)
        - vix: VIX level — GMM regime detection input only, not in feature matrix X
        - 11 stationary technical features:
            ret_5d, ret_21d, vol_20d, atr_14, rsi_14,
            macd_line, macd_signal, macd_hist, bb_pct, bb_width, volume_ratio
        - Triple-barrier labels:
            label (ternary: +1/−1/0),
            label_binary (+1 = take profit, −1 = stop/time barrier)
        - Forward returns for backtesting:
            ret_1d_forward (next-day P&L for trading signals),
            ret_10d_forward (10-day return, IC target for feature selection)

        Returns MultiIndex (ticker, date) DataFrame.
        """
        return self.load_from_hdf5(key=self.features_key)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw aligned data (OHLCV + VIX, no features or labels)."""
        return self.load_from_hdf5(key=self.raw_data_key)

    # ------------------------------------------------------------------
    # Inspection and validation utilities
    # ------------------------------------------------------------------

    def list_hdf5_keys(self) -> List[str]:
        """Return all dataset keys in the HDF5 file (e.g. ['/data_raw', '/engineered_features'])."""
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        try:
            with pd.HDFStore(self.hdf5_path, mode='r') as store:
                keys = list(store.keys())
            logger.info(f"HDF5 keys: {keys}")
            return keys
        except Exception as e:
            raise ValueError(f"Failed to list HDF5 keys: {e}") from e

    def get_hdf5_info(self) -> Dict[str, Any]:
        """Return file size and per-key metadata (shape, columns, memory)."""
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        try:
            file_size_mb = self.hdf5_path.stat().st_size / (1024 * 1024)

            with pd.HDFStore(self.hdf5_path, mode='r') as store:
                keys = list(store.keys())
                info: Dict[str, Any] = {
                    'file_path': str(self.hdf5_path),
                    'file_size_mb': round(file_size_mb, 2),
                    'num_keys': len(keys),
                    'keys': keys,
                    'key_info': {}
                }

                for key in keys:
                    df = store.get(key)
                    info['key_info'][key] = {
                        'shape': df.shape,
                        'columns': df.columns.tolist() if hasattr(df, 'columns') else None,
                        'index_type': type(df.index).__name__,
                        'memory_mb': round(
                            df.memory_usage(deep=True).sum() / (1024 * 1024), 2
                        )
                    }

            return info

        except Exception as e:
            raise ValueError(f"Failed to read HDF5 metadata: {e}") from e

    def validate_hdf5_data(self, key: Optional[str] = None) -> bool:
        """
        Validate structure of a stored DataFrame.
        Checks: non-empty, MultiIndex (ticker, date), missing value count.
        Missing values are reported as warnings (not raised) to allow partial inspection.
        """
        key = key or self.raw_data_key
        logger.info(f"Validating HDF5 data for key '{key}'")

        data = self.load_from_hdf5(key)

        if data.empty:
            raise ValueError(f"Data for key '{key}' is empty")

        if not isinstance(data.index, pd.MultiIndex):
            logger.warning(
                f"Key '{key}': expected MultiIndex, got {type(data.index).__name__}"
            )
        elif list(data.index.names) != ['ticker', 'date']:
            logger.warning(
                f"Key '{key}': expected index names ['ticker', 'date'], "
                f"got {data.index.names}"
            )

        null_counts = data.isnull().sum()
        missing = int(null_counts.sum())
        if missing > 0:
            logger.warning(f"Key '{key}': {missing} NaN values")
            for col, n in null_counts[null_counts > 0].items():
                logger.debug(f"  {col}: {n} NaN")
        else:
            logger.info(f"Key '{key}': no missing values")

        logger.success(f"Validation passed for key '{key}'")
        return True

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return shape, columns, missing values, memory footprint, and date range."""
        summary: Dict[str, Any] = {
            'shape': data.shape,
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'columns': data.columns.tolist(),
            'missing_values': int(data.isnull().sum().sum()),
            'memory_mb': round(
                data.memory_usage(deep=True).sum() / (1024 * 1024), 2
            )
        }

        if isinstance(data.index, pd.MultiIndex):
            summary['index_type'] = 'MultiIndex'
            summary['index_names'] = data.index.names
            if 'date' in data.index.names:
                dates = data.index.get_level_values('date')
                summary['date_range'] = {'start': str(dates.min()), 'end': str(dates.max())}
        elif isinstance(data.index, pd.DatetimeIndex):
            summary['index_type'] = 'DatetimeIndex'
            summary['date_range'] = {
                'start': str(data.index.min()),
                'end': str(data.index.max())
            }
        else:
            summary['index_type'] = type(data.index).__name__

        return summary
