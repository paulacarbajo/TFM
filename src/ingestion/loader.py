"""
Data Loader Module

Loads and saves financial data using HDF5 format.
Uses pandas HDFStore with blosc compression for efficient storage
and retrieval of large DataFrames with MultiIndex structure.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from loguru import logger

# Project root — two levels up from src/ingestion/loader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DataLoader:
    """
    Loads and saves financial data using HDF5 format.

    The HDF5 file stores two datasets under separate keys:
    - data_raw: aligned OHLCV + FRED data before feature engineering
    - engineered_features: full dataset with technical features and labels

    Using format='table' allows column-level querying and is required for
    DataFrames with MultiIndex. Compression level 9 with blosc provides
    a good balance between file size and read/write speed.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        hdf5_path (Path): Path to HDF5 file
        raw_data_key (str): HDF5 key for raw data
        features_key (str): HDF5 key for engineered features
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        ingestion_config = config.get('ingestion', {})

        hdf5_file = ingestion_config.get('hdf5_file', 'data/processed/assets.h5')
        self.hdf5_path = PROJECT_ROOT / hdf5_file
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        hdf5_keys = ingestion_config.get('hdf5_keys', {})
        self.raw_data_key = hdf5_keys.get('raw_data', 'data_raw')
        self.features_key = hdf5_keys.get('engineered_features', 'engineered_features')

        logger.info(f"DataLoader initialized with HDF5 file: {self.hdf5_path}")
        logger.info(
            f"Keys: raw_data='{self.raw_data_key}', "
            f"features='{self.features_key}'"
        )

    def save_to_hdf5(
        self,
        data: pd.DataFrame,
        key: Optional[str] = None,
        mode: str = 'a',
        complevel: int = 9,
        complib: str = 'blosc'
    ) -> None:
        """
        Save DataFrame to HDF5 file.

        Uses format='table' to support MultiIndex DataFrames and column-level
        queries. data_columns=True makes all columns indexable.

        Args:
            data: DataFrame to save
            key: HDF5 key (defaults to raw_data_key)
            mode: 'a' to append/update, 'w' to overwrite the entire file
            complevel: Compression level 0-9 (9 = maximum compression)
            complib: Compression library ('blosc' is fastest for read/write)

        Raises:
            ValueError: If save fails
        """
        key = key or self.raw_data_key

        logger.info(f"Saving data to HDF5: {self.hdf5_path} with key '{key}'")
        logger.info(f"Data shape: {data.shape}")

        try:
            with pd.HDFStore(
                self.hdf5_path,
                mode=mode,
                complevel=complevel,
                complib=complib
            ) as store:
                store.put(key, data, format='table', data_columns=True)

            file_size_mb = self.hdf5_path.stat().st_size / (1024 * 1024)
            logger.success(
                f"Saved to HDF5 successfully. "
                f"File size: {file_size_mb:.2f} MB"
            )

        except Exception as e:
            logger.error(f"Failed to save to HDF5: {str(e)}")
            raise ValueError(f"HDF5 save failed: {str(e)}")

    def load_from_hdf5(self, key: Optional[str] = None) -> pd.DataFrame:
        """
        Load DataFrame from HDF5 file.

        Args:
            key: HDF5 key to load (defaults to raw_data_key)

        Returns:
            DataFrame loaded from HDF5

        Raises:
            FileNotFoundError: If HDF5 file does not exist
            ValueError: If key does not exist or load fails
        """
        key = key or self.raw_data_key

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        logger.info(f"Loading from HDF5: {self.hdf5_path} with key '{key}'")

        try:
            with pd.HDFStore(self.hdf5_path, mode='r') as store:
                if key not in store:
                    available_keys = list(store.keys())
                    raise ValueError(
                        f"Key '{key}' not found in HDF5. "
                        f"Available keys: {available_keys}"
                    )
                data = store.get(key)

            logger.success(f"Loaded data with shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load from HDF5: {str(e)}")
            raise ValueError(f"HDF5 load failed: {str(e)}")

    def list_hdf5_keys(self) -> List[str]:
        """
        List all keys stored in the HDF5 file.

        Returns:
            List of key strings

        Raises:
            FileNotFoundError: If HDF5 file does not exist
        """
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        try:
            with pd.HDFStore(self.hdf5_path, mode='r') as store:
                keys = list(store.keys())

            logger.info(f"HDF5 keys: {keys}")
            return keys

        except Exception as e:
            logger.error(f"Failed to list HDF5 keys: {str(e)}")
            raise ValueError(f"Failed to list keys: {str(e)}")

    def get_hdf5_info(self) -> Dict[str, Any]:
        """
        Get metadata about the HDF5 file and its contents.

        Returns:
            Dictionary with file size, keys, shapes, and memory usage

        Raises:
            FileNotFoundError: If HDF5 file does not exist
        """
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        try:
            file_size_mb = self.hdf5_path.stat().st_size / (1024 * 1024)

            with pd.HDFStore(self.hdf5_path, mode='r') as store:
                keys = list(store.keys())
                info = {
                    'file_path': str(self.hdf5_path),
                    'file_size_mb': round(file_size_mb, 2),
                    'num_keys': len(keys),
                    'keys': keys,
                    'key_info': {}
                }

                for key in keys:
                    data = store.get(key)
                    info['key_info'][key] = {
                        'shape': data.shape,
                        'columns': (
                            data.columns.tolist()
                            if hasattr(data, 'columns') else None
                        ),
                        'index_type': type(data.index).__name__,
                        'memory_mb': round(
                            data.memory_usage(deep=True).sum() / (1024 * 1024), 2
                        )
                    }

            return info

        except Exception as e:
            logger.error(f"Failed to get HDF5 info: {str(e)}")
            raise ValueError(f"Failed to get info: {str(e)}")

    def validate_hdf5_data(self, key: Optional[str] = None) -> bool:
        """
        Validate data structure in HDF5 file.

        Args:
            key: HDF5 key to validate (defaults to raw_data_key)

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        key = key or self.raw_data_key

        logger.info(f"Validating HDF5 data for key '{key}'")

        data = self.load_from_hdf5(key)

        if data.empty:
            raise ValueError(f"Data for key '{key}' is empty")

        if not isinstance(data.index, pd.MultiIndex):
            logger.warning(f"Data for key '{key}' does not have MultiIndex")
        elif data.index.names != ['ticker', 'date']:
            logger.warning(
                f"MultiIndex names are {data.index.names}, "
                f"expected ['ticker', 'date']"
            )

        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Data contains {missing_count} missing values")

        logger.success(f"Validation passed for key '{key}'")
        return True

    def load_engineered_features(self) -> pd.DataFrame:
        """Load engineered features from HDF5."""
        return self.load_from_hdf5(key=self.features_key)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw aligned data from HDF5."""
        return self.load_from_hdf5(key=self.raw_data_key)

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for a DataFrame.

        Args:
            data: DataFrame to summarize

        Returns:
            Dictionary with shape, columns, missing values, and date range
        """
        summary = {
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
            summary['index_levels'] = {
                name: data.index.get_level_values(name).unique().tolist()
                for name in data.index.names
            }
        else:
            summary['index_type'] = type(data.index).__name__

        if isinstance(data.index, pd.DatetimeIndex):
            summary['date_range'] = {
                'start': str(data.index.min()),
                'end': str(data.index.max())
            }
        elif isinstance(data.index, pd.MultiIndex) and 'date' in data.index.names:
            date_level = data.index.get_level_values('date')
            summary['date_range'] = {
                'start': str(date_level.min()),
                'end': str(date_level.max())
            }

        return summary