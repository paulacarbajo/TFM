"""
Walk-Forward Cross-Validation with Regime Detection (Iteration 2)

Extends the baseline walk-forward by adding GMM-based regime features
to the training and validation sets at each fold. The GMM is fitted
exclusively on training data to avoid look-ahead bias.

Regime features added per fold:
- regime_state: integer in {0, 1, 2} ordered by volatility
- regime_prob_0: probability of low-volatility regime
- regime_prob_1: probability of medium-volatility regime
- regime_prob_2: probability of high-volatility regime

Results are saved to data/processed/walk_forward_results_regime.pkl.
"""

import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger

from src.ingestion.loader import DataLoader
from src.models import ModelTrainer, WalkForwardCV
from src.models.regime_detection import RegimeDetector

REGIME_COLS = [
    'regime_state',
    'regime_prob_0',
    'regime_prob_1',
    'regime_prob_2'
]


def main():
    """Run walk-forward cross-validation with regime detection."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    logger.add(
        "logs/walk_forward_regime_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("=" * 80)
    logger.info("WALK-FORWARD CROSS-VALIDATION WITH REGIME DETECTION")
    logger.info("=" * 80)

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    logger.info("Loading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()

    logger.info(f"Data loaded: {data.shape}")
    logger.info(
        f"Date range: {data.index.get_level_values('date').min()} to "
        f"{data.index.get_level_values('date').max()}"
    )

    # Initialize components
    wf_cv = WalkForwardCV(config)
    trainer = ModelTrainer(config)

    all_fold_results = []

    for fold_data in wf_cv.split(data):
        fold_number = fold_data['fold_number']

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            f"PROCESSING FOLD {fold_number} "
            f"WITH REGIME DETECTION"
        )
        logger.info("=" * 80)

        X_train = fold_data['X_train']
        X_val = fold_data['X_val']
        
        # Get full DataFrames with all columns (including vix for regime detection)
        train_data_full = fold_data['train_data_full']
        val_data_full = fold_data['val_data_full']

        logger.info(f"Original features: {X_train.shape[1]}")

        # Step 1: Fit GMM on train, infer on both train and val
        # Use full DataFrames that include vix for regime detection
        logger.info("Detecting market regimes...")
        regime_detector = RegimeDetector(config, n_regimes=3)
        train_data_with_regime, val_data_with_regime = regime_detector.fit_predict(
            train_data_full, val_data_full
        )
        
        # Step 2: Extract regime features and add them to X_train and X_val
        regime_features = ['regime_state', 'regime_prob_0', 'regime_prob_1', 'regime_prob_2']
        
        # Align regime features with X_train and X_val indices
        X_train_r = X_train.copy()
        X_val_r = X_val.copy()
        
        for feat in regime_features:
            if feat in train_data_with_regime.columns:
                X_train_r[feat] = train_data_with_regime.loc[X_train.index, feat]
            if feat in val_data_with_regime.columns:
                X_val_r[feat] = val_data_with_regime.loc[X_val.index, feat]

        logger.info(
            f"Fold {fold_number}: {X_train.shape[1]} features → "
            f"{X_train_r.shape[1]} "
            f"(+{X_train_r.shape[1] - X_train.shape[1]} regime features)"
        )

        # Step 2: Update fold data with regime-enriched features
        fold_data_regime = fold_data.copy()
        fold_data_regime['X_train'] = X_train_r
        fold_data_regime['X_val'] = X_val_r
        fold_data_regime['feature_names'] = (
            fold_data['feature_names'] + REGIME_COLS
        )

        logger.info(
            f"Total features for training: "
            f"{len(fold_data_regime['feature_names'])}"
        )

        # Step 3: Train models with regime features
        logger.info("Training models with regime features...")
        fold_results = trainer.train_fold(fold_data_regime)
        fold_results['regime_detector'] = regime_detector

        all_fold_results.append(fold_results)
        logger.info(f"Fold {fold_number} complete")

    # Save results (evaluation will be done in backtest)
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    output_path = Path('data/processed/walk_forward_results_regime.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'all_fold_results': all_fold_results
        }, f)

    logger.info(f"Results saved to {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD WITH REGIME DETECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total folds: {len(all_fold_results)}")
    logger.info(f"Models trained: LightGBM, EBM")
    logger.info(f"Regime features: {len(REGIME_COLS)} columns added per fold")
    logger.info(f"Results saved: {output_path}")
    logger.info(f"Note: Evaluation will be performed during backtesting")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
