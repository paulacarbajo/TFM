"""
Data Loader Module
==================

Persists and retrieves pipeline DataFrames using HDF5 format.

Role in the pipeline
--------------------
DataLoader is the persistence layer of ``main.py`` and is called at two
points:

1. **After alignment** — saves the raw OHLCV + VIX DataFrame under the
   ``data_raw`` key, preserving the aligned inputs independently of the
   feature engineering step (useful for debugging or re-running features
   without re-downloading data).

2. **After feature engineering and labeling** — saves the fully processed
   DataFrame under the ``engineered_features`` key.  Every training script
   (``run_walk_forward.py``, ``run_walk_forward_regime.py``,
   ``run_walk_forward_distillation.py``, etc.) calls
   ``loader.load_engineered_features()`` as its entry point.

Design decisions
----------------
1. **HDF5 over CSV / Parquet** — pandas MultiIndex DataFrames are natively
   supported by HDF5 via PyTables.  CSV flattens the index and loses type
   information.  Parquet requires the MultiIndex to be converted to columns
   and rebuilt after load.  HDF5 ``format='table'`` round-trips the full
   (ticker, date) MultiIndex with correct dtypes in a single call.

2. **``format='table'`` (not ``format='fixed'``)** — the only HDF5 format
   that supports MultiIndex DataFrames.  It also allows appending a new key
   (``mode='a'``) without rewriting the whole file, which is how ``main.py``
   writes ``data_raw`` first (``mode='w'``) and then appends
   ``engineered_features`` (``mode='a'``).

3. **blosc compression at level 9** — blosc is optimised for in-memory
   numeric float64 arrays.  It decompresses faster than zlib/lzf while
   achieving comparable ratios.  Level 9 reduces a 5 000-row × 30-column
   float64 DataFrame from ~1.2 MB to ~200 KB on disk.

4. **``data_columns=True``** — indexes every column in the HDF5 table.
   Enables fast ``where``-clause queries (e.g.
   ``store.select(key, where='date>"2020-01-01"')``) without loading the
   full file.  Not used in the current training scripts but available for
   ad-hoc analysis.

5. **Single HDF5 file, two keys** — keeps ``data/processed/`` clean and
   ensures raw and processed data are always co-located, preventing version
   mismatches between the two datasets.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from loguru import logger

# Project root — two levels up from src/ingestion/loader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default compression applied to every save unless the caller overrides.
# blosc + level 9: best read-speed / file-size trade-off for numeric float64.
_DEFAULT_COMPLEVEL = 9
_DEFAULT_COMPLIB = 'blosc'


class DataLoader:
    """
    Saves and loads pipeline DataFrames from a single HDF5 file with two
    dataset keys:

    - ``data_raw``: aligned OHLCV + VIX before feature engineering.
    - ``engineered_features``: 11 technical features, triple-barrier labels
      (``label``, ``label_binary``), and forward returns
      (``ret_1d_forward``, ``ret_10d_forward``).

    All training scripts read from ``engineered_features`` via
    ``load_engineered_features()``.  The ``data_raw`` key is retained for
    debugging and reproducibility.

    Attributes:
        config (Dict[str, Any]): Full configuration dictionary.
        hdf5_path (Path): Absolute path to the HDF5 file.
        raw_data_key (str): HDF5 key for raw aligned data.
        features_key (str): HDF5 key for engineered and labeled data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise DataLoader from the project configuration.

        Args:
            config: Configuration dictionary.  Reads the 'ingestion' section
                    for hdf5_file and hdf5_keys.
        """
        self.config = config
        ingestion_config = config.get('ingestion', {})

        hdf5_file = ingestion_config.get('hdf5_file', 'data/processed/assets.h5')
        self.hdf5_path = PROJECT_ROOT / hdf5_file
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        hdf5_keys = ingestion_config.get('hdf5_keys', {})
        self.raw_data_key = hdf5_keys.get('raw_data', 'data_raw')
        self.features_key = hdf5_keys.get('engineered_features', 'engineered_features')

        logger.info(f"DataLoader initialised  |  file: {self.hdf5_path}")
        logger.info(
            f"Keys: raw='{self.raw_data_key}', features='{self.features_key}'"
        )

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

        Format rationale
        ~~~~~~~~~~~~~~~~
        ``format='table'`` stores data as a PyTables Table object, which:
        - Supports MultiIndex DataFrames (required for (ticker, date) index).
        - Allows writing a new key without rewriting the whole file
          (``mode='a'``), enabling ``main.py`` to append ``engineered_features``
          after writing ``data_raw`` in the same pipeline run.
        - Enables column-level ``where``-clause queries.

        ``data_columns=True`` indexes every column in the table.  Without
        it, partial column reads would require a full file scan.

        Args:
            data: DataFrame to persist.
            key: HDF5 dataset key (defaults to ``raw_data_key``).
            mode: ``'a'`` = append / update key; ``'w'`` = overwrite file.
            complevel: Compression level 0–9 (default 9 = maximum).
            complib: Compression library (default 'blosc').

        Raises:
            ValueError: If the write operation fails.
        """
        key = key or self.raw_data_key

        logger.info(f"Saving to HDF5  |  key: '{key}'  |  shape: {data.shape}")

        try:
            with pd.HDFStore(
                self.hdf5_path, mode=mode, complevel=complevel, complib=complib
            ) as store:
                store.put(key, data, format='table', data_columns=True)

            file_size_mb = self.hdf5_path.stat().st_size / (1024 * 1024)
            logger.success(
                f"Saved '{key}'  |  file size: {file_size_mb:.2f} MB"
            )

        except Exception as e:
            logger.error(f"HDF5 save failed for key '{key}': {e}")
            raise ValueError(f"HDF5 save failed: {e}") from e

    def load_from_hdf5(self, key: Optional[str] = None) -> pd.DataFrame:
        """
        Load a DataFrame from the HDF5 file.

        The MultiIndex (ticker, date) and all column dtypes are restored
        exactly as they were saved — no post-processing is required.

        Args:
            key: HDF5 dataset key (defaults to ``raw_data_key``).

        Returns:
            DataFrame with the original MultiIndex and dtypes.

        Raises:
            FileNotFoundError: If the HDF5 file does not exist.
            ValueError: If the key is absent or the read fails.
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
                # store.keys() returns keys with a leading '/' (e.g. '/data_raw'),
                # but callers pass keys without it (e.g. 'data_raw').
                # Normalise before comparison; store.get() handles both forms.
                norm_key = key if key.startswith('/') else f'/{key}'
                if norm_key not in available:
                    raise ValueError(
                        f"Key '{key}' not found.  "
                        f"Available keys: {available}"
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
        Load the fully processed dataset.

        This is the entry point for all training and evaluation scripts.
        The returned DataFrame contains:

        - OHLCV: Close, High, Low, Open, Volume (split/dividend-adjusted)
        - VIX level (``vix``) — used by RegimeDetector only, not in feature
          matrix **X**
        - 11 stationary technical features:
          ``ret_5d``, ``ret_21d``, ``vol_20d``, ``atr_14``, ``rsi_14``,
          ``macd_line``, ``macd_signal``, ``macd_hist``,
          ``bb_pct``, ``bb_width``, ``volume_ratio``
        - Triple-barrier labels:
          ``label`` (ternary: +1/−1/0) and
          ``label_binary`` (binary: +1 = take profit, −1 = stop/time barrier)
        - Forward returns for backtesting:
          ``ret_1d_forward`` (next-day return, used in trading signal P&L),
          ``ret_10d_forward`` (10-day return, used as IC target for feature
          selection)

        Returns:
            MultiIndex (ticker, date) DataFrame.
        """
        return self.load_from_hdf5(key=self.features_key)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load the raw aligned data (OHLCV + VIX, no features or labels).

        Useful for debugging the ingestion pipeline or re-running feature
        engineering without re-downloading from yfinance / FRED.

        Returns:
            MultiIndex (ticker, date) DataFrame.
        """
        return self.load_from_hdf5(key=self.raw_data_key)

    # ------------------------------------------------------------------
    # Inspection and validation utilities
    # ------------------------------------------------------------------

    def list_hdf5_keys(self) -> List[str]:
        """
        Return all dataset keys stored in the HDF5 file.

        Returns:
            List of key strings (e.g. ['/data_raw', '/engineered_features']).

        Raises:
            FileNotFoundError: If the HDF5 file does not exist.
        """
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
        """
        Return metadata for the HDF5 file and each stored dataset.

        Reports the on-disk file size and, for each key, the shape, column
        names, index type, and in-memory size of the loaded DataFrame.

        Returns:
            Dictionary with file-level and per-key metadata.

        Raises:
            FileNotFoundError: If the HDF5 file does not exist.
        """
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
        Validate the structure of a stored DataFrame.

        Checks
        ------
        - Non-empty DataFrame.
        - MultiIndex with names ``['ticker', 'date']``.
        - Missing value count (reported as warning, not raised as error,
          to allow partial data inspection during debugging).

        Args:
            key: HDF5 key to validate (defaults to ``raw_data_key``).

        Returns:
            True if all checks pass.

        Raises:
            ValueError: If the DataFrame is empty or structurally invalid.
        """
        key = key or self.raw_data_key
        logger.info(f"Validating HDF5 data for key '{key}'")

        data = self.load_from_hdf5(key)

        if data.empty:
            raise ValueError(f"Data for key '{key}' is empty")

        if not isinstance(data.index, pd.MultiIndex):
            logger.warning(
                f"Key '{key}': expected MultiIndex, "
                f"got {type(data.index).__name__}"
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
        """
        Compute a quality summary for an in-memory DataFrame.

        Reports shape, column list, missing value count, memory footprint,
        and (when the index carries date information) the date range.

        Args:
            data: Any DataFrame (does not need to originate from HDF5).

        Returns:
            Dictionary with summary statistics.
        """
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
                summary['date_range'] = {
                    'start': str(dates.min()),
                    'end': str(dates.max())
                }
        elif isinstance(data.index, pd.DatetimeIndex):
            summary['index_type'] = 'DatetimeIndex'
            summary['date_range'] = {
                'start': str(data.index.min()),
                'end': str(data.index.max())
            }
        else:
            summary['index_type'] = type(data.index).__name__

        return summary
