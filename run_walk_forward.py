"""
Run complete walk-forward cross-validation.

Trains LightGBM and EBM on all folds and evaluates performance.
Results are saved to data/processed/walk_forward_results.pkl.
"""

import argparse
import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger

from src.ingestion.loader import DataLoader
from src.models import ModelTrainer


def main():
    """Run complete walk-forward cross-validation."""
    parser = argparse.ArgumentParser(description='Walk-forward cross-validation')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to config YAML (default: config/config.yaml)')
    args = parser.parse_args()

    # Derive output suffix from config stem: 'config' → '', 'config_2010' → '_2010'
    config_stem = Path(args.config).stem
    suffix = config_stem[len('config'):]

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    logger.add(
        "logs/walk_forward_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("=" * 80)
    logger.info("STARTING WALK-FORWARD CROSS-VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}  (output suffix: '{suffix}')")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    logger.info("\n1. Loading engineered features...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()
    logger.info(f"Data loaded: {data.shape}")

    # Train all folds
    logger.info("\n2. Training all folds...")
    trainer = ModelTrainer(config)
    all_fold_results = trainer.train_all_folds(data)
    logger.info(f"Training complete: {len(all_fold_results)} folds")

    # Save results — used by run_shap_analysis.py and run_rolling_oos_evaluation.py
    logger.info("\n3. Saving results...")
    output_path = Path(f'data/processed/walk_forward_results{suffix}.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'all_fold_results': all_fold_results
        }, f)

    logger.info(f"Results saved to {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD CROSS-VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total folds: {len(all_fold_results)}")
    logger.info(f"Models trained: LightGBM (EBM Distilled and RuleFit applied separately)")
    logger.info(f"Results saved: {output_path}")
    logger.info(f"Next steps: run_shap_analysis.py (fold SHAP) · run_rolling_oos_evaluation.py (quarterly backtest)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
