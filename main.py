"""
Main Entry Point for Data Ingestion and Feature Engineering Pipeline

Orchestrates the full data preparation pipeline:
1. Download market data from yfinance (SPY, 2004-2024)
2. Download macroeconomic data from FRED (VIX / VIXCLS)
3. Align yfinance and FRED data via inner join on trading dates
4. Save raw aligned data to HDF5
5. Calculate 11 stationary technical features (momentum, volatility, RSI,
   normalized MACD, Bollinger %B and width, normalized ATR%, volume ratio,
   trend distance)
6. Apply triple barrier labeling (ternary label + binary collapse)
7. Save final labeled dataset to HDF5

Output: data/processed/assets.h5 with two keys:
- data_raw: aligned OHLCV + FRED data before feature engineering
- engineered_features: full dataset with technical features and labels
"""

import sys
import warnings
from pathlib import Path

import yaml
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import DataDownloader, FREDLoader, DataAligner, DataLoader
from src.features import FeatureEngineer, TripleBarrierLabeler


def setup_logging() -> None:
    """
    Configure loguru logging to console and rotating file.

    Console: INFO level with colour formatting.
    File: DEBUG level, daily rotation, 30-day retention.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level="INFO"
    )
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/pipeline_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG"
    )


def print_summary(summary: dict) -> None:
    """
    Log alignment summary in a structured format.

    Args:
        summary: Dictionary from DataAligner.get_alignment_summary()
    """
    logger.info("=" * 80)
    logger.info("DATA ALIGNMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Input yfinance rows:  {summary['input_yfinance_rows']}")
    logger.info(f"Input FRED rows:      {summary['input_fred_rows']}")
    logger.info(f"Output rows:          {summary['output_rows']}")
    logger.info(f"Tickers:              {', '.join(summary['tickers'])}")
    logger.info(f"Total columns:        {summary['total_columns']}")
    logger.info(
        f"Date range:           {summary['date_range']['start']} "
        f"to {summary['date_range']['end']}"
    )
    logger.info(f"Missing values:       {summary['missing_values']}")
    logger.info("")
    logger.info("Rows per ticker:")
    for ticker, rows in summary['rows_per_ticker'].items():
        logger.info(f"  {ticker}: {rows} rows")
    logger.info("")
    logger.info(
        f"yfinance columns ({len(summary['yfinance_columns'])}): "
        f"{summary['yfinance_columns']}"
    )
    logger.info(
        f"FRED columns ({len(summary['fred_columns'])}): "
        f"{summary['fred_columns'][:10]}..."
    )
    logger.info("=" * 80)


def main() -> int:
    """
    Execute the full data ingestion and feature engineering pipeline.

    Returns:
        0 on success, 1 on failure
    """
    setup_logging()

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    logger.info("=" * 80)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("=" * 80)

    try:
        # Load configuration
        config_path = Path('config/config.yaml')
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info("Configuration loaded successfully")

        # ------------------------------------------------------------------
        # Step 1: Download yfinance data
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 1: DOWNLOADING YFINANCE DATA")
        logger.info("=" * 80)

        downloader = DataDownloader(config)
        yfinance_data = downloader.download_all_assets()

        yf_summary = downloader.get_data_summary(yfinance_data)
        logger.info(
            f"Downloaded {yf_summary['total_rows']} rows "
            f"for {yf_summary['num_tickers']} tickers"
        )

        # ------------------------------------------------------------------
        # Step 2: Download FRED data
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 2: DOWNLOADING FRED DATA")
        logger.info("=" * 80)

        fred_loader = FREDLoader(config)
        fred_data = fred_loader.prepare_fred_data()

        fred_summary = fred_loader.get_data_summary(fred_data)
        logger.info(
            f"Downloaded {fred_summary['num_series']} FRED series, "
            f"{fred_summary['total_columns']} total columns"
        )

        # ------------------------------------------------------------------
        # Step 3: Align yfinance and FRED data
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 3: ALIGNING YFINANCE AND FRED DATA")
        logger.info("=" * 80)

        aligner = DataAligner(config)
        aligned_data = aligner.align_yfinance_with_fred(yfinance_data, fred_data)
        aligner.validate_alignment(aligned_data)

        # ------------------------------------------------------------------
        # Step 4: Save raw aligned data to HDF5
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 4: SAVING RAW DATA TO HDF5")
        logger.info("=" * 80)

        loader = DataLoader(config)
        loader.save_to_hdf5(aligned_data, mode='w')

        hdf5_info = loader.get_hdf5_info()
        logger.info(f"HDF5 file size: {hdf5_info['file_size_mb']} MB")
        loader.validate_hdf5_data()

        # ------------------------------------------------------------------
        # Alignment summary (informational, not a pipeline step)
        # ------------------------------------------------------------------
        logger.info("")
        summary = aligner.get_alignment_summary(yfinance_data, fred_data, aligned_data)
        print_summary(summary)

        logger.info("Sample of aligned data (first 5 rows):")
        logger.info(f"\n{aligned_data.head(5)}")
        logger.info("Sample of aligned data (last 5 rows):")
        logger.info(f"\n{aligned_data.tail(5)}")
        logger.info(f"Total columns: {len(aligned_data.columns)}")

        # ------------------------------------------------------------------
        # Step 5: Feature engineering
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 5: FEATURE ENGINEERING")
        logger.info("=" * 80)

        engineer = FeatureEngineer(config)
        featured_data = engineer.engineer_features(aligned_data)

        feature_summary = engineer.get_feature_summary(featured_data)
        logger.info(f"Technical features added: {feature_summary['active_feature_count']}")
        logger.info(f"Total columns: {feature_summary['total_columns']}")

        # ------------------------------------------------------------------
        # Step 6: Triple barrier labeling
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 6: TRIPLE BARRIER LABELING")
        logger.info("=" * 80)

        labeler = TripleBarrierLabeler(config)
        labeled_data = labeler.label_data(featured_data)

        label_summary = labeler.get_label_summary(labeled_data)
        logger.info(f"Total observations:      {label_summary['total_observations']}")
        logger.info(f"Ternary distribution:    {label_summary['ternary_label_distribution']}")
        logger.info(f"Binary distribution:     {label_summary['binary_label_distribution']}")
        logger.info(f"Missing labels:          {label_summary['missing_labels']}")

        # ------------------------------------------------------------------
        # Step 7: Save engineered features to HDF5
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 7: SAVING ENGINEERED FEATURES TO HDF5")
        logger.info("=" * 80)

        loader.save_to_hdf5(
            labeled_data,
            key=loader.features_key,
            mode='a'
        )

        logger.success(
            f"Engineered features saved with key '{loader.features_key}'"
        )

        # ------------------------------------------------------------------
        # Step 8: Final verification
        # ------------------------------------------------------------------
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 8: FINAL VERIFICATION")
        logger.info("=" * 80)

        all_keys = loader.list_hdf5_keys()
        final_info = loader.get_hdf5_info()

        logger.info(f"HDF5 keys: {all_keys}")
        logger.info(f"HDF5 file size: {final_info['file_size_mb']} MB")

        for key, info in final_info['key_info'].items():
            logger.info(f"\nKey '{key}':")
            logger.info(f"  Shape:   {info['shape']}")
            logger.info(f"  Memory:  {info['memory_mb']} MB")
            logger.info(
                f"  Columns: {len(info['columns']) if info['columns'] else 'N/A'}"
            )

        logger.info("")
        logger.info("Sample of final labeled data (first 5 rows):")
        logger.info(f"\n{labeled_data.head(5)}")

        logger.info("")
        logger.info("=" * 80)
        logger.success("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"HDF5 file:     {loader.hdf5_path}")
        logger.info(f"Keys:          {all_keys}")
        logger.info(f"Final shape:   {labeled_data.shape}")
        logger.info(
            f"Features:      {feature_summary['active_feature_count']} stationary "
            f"technical indicators (11 baseline)"
        )
        logger.info(
            "Labels:        label (ternary: 1/-1/0) and "
            "label_binary (binary: 1/-1 → converted to 0/1 at training)"
        )
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.exception("Full traceback:")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())