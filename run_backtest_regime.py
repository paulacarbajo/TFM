"""
Backtesting with Regime Detection (Iteration 2)

Runs backtesting on the OOS period 2020-2024 using models trained with
regime features. The regime detector from fold 8 is reused to add regime
features to the OOS period, ensuring feature consistency with training.

No look-ahead bias is introduced because the fold 8 regime detector was
fitted exclusively on data up to 2019.

Results are saved to data/processed/backtest_results_regime.pkl.
"""

import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger

from src.ingestion.loader import DataLoader
from src.models.backtest import Backtester
from src.models.regime_detection import RegimeDetector

REGIME_COLS = [
    'regime_state',
    'regime_prob_0',
    'regime_prob_1',
    'regime_prob_2'
]


def main():
    """Run backtesting with regime features on OOS period."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    logger.add(
        "logs/backtest_regime_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("=" * 80)
    logger.info("BACKTESTING WITH REGIME DETECTION")
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

    # Load walk-forward regime results
    wf_path = Path('data/processed/walk_forward_results_regime.pkl')
    if not wf_path.exists():
        logger.error(f"Walk-forward regime results not found at {wf_path}")
        logger.error(
            "Please run walk-forward with regime first: "
            "python run_walk_forward_regime.py"
        )
        return

    logger.info(f"Loading walk-forward regime results from {wf_path}...")
    with open(wf_path, 'rb') as f:
        wf_results = pickle.load(f)

    all_fold_results = wf_results['all_fold_results']
    logger.info(f"Loaded {len(all_fold_results)} fold results")

    # Retrieve regime detector from last fold
    # This detector was fitted on data up to 2019 — no look-ahead bias
    last_fold = all_fold_results[-1]
    regime_detector = last_fold.get('regime_detector')

    if regime_detector is None:
        logger.error("No regime detector found in fold 8 results")
        logger.error(
            "The walk-forward results may be from an old run. "
            "Please re-run: python run_walk_forward_regime.py"
        )
        return

    logger.info(
        f"Regime detector loaded from fold {last_fold['fold_number']} "
        f"({regime_detector.n_regimes} regimes)"
    )

    # Filter OOS period
    oos_start = (
        config.get('models', {})
        .get('walk_forward', {})
        .get('test_start', '2020-01-01')
    )
    oos_end = (
        config.get('models', {})
        .get('walk_forward', {})
        .get('test_end', '2024-12-31')
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("ADDING REGIME FEATURES TO OOS PERIOD")
    logger.info("=" * 80)
    logger.info(f"OOS period: {oos_start} to {oos_end}")

    dates = data.index.get_level_values('date')
    mask = (dates >= oos_start) & (dates <= oos_end)
    data_oos = data[mask].copy()

    logger.info(f"OOS data: {data_oos.shape}")

    # Extract feature columns (excluding OHLCV and labels)
    exclude_cols = config.get('models', {}).get('exclude_from_features', [])
    X_oos = data_oos.drop(columns=exclude_cols, errors='ignore')

    logger.info(f"OOS features before regime: {X_oos.shape[1]}")

    # Apply fold 8 regime detector to OOS data (inference only, no retraining)
    regime_state, regime_proba = regime_detector.predict(X_oos)

    if regime_state is None:
        logger.error("Regime prediction failed for OOS period — cannot proceed")
        return

    # Add regime columns to OOS dataset
    data_oos_regime = regime_detector.add_regime_features(
        data_oos, regime_state, regime_proba
    )

    logger.info(f"OOS features after regime: {data_oos_regime.shape[1]}")

    # Log regime distribution in OOS
    logger.info("\nRegime distribution in OOS period:")
    regime_dist = data_oos_regime['regime_state'].value_counts().sort_index()
    for regime, count in regime_dist.items():
        pct = count / len(data_oos_regime) * 100
        logger.info(f"  Regime {int(regime)}: {count} observations ({pct:.1f}%)")

    # Run backtest
    logger.info("")
    logger.info("=" * 80)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 80)

    backtester = Backtester(config)
    backtest_results = backtester.run_backtest(all_fold_results, data_oos_regime)

    # Results table
    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS TABLE")
    logger.info("=" * 80)

    table = backtester.get_backtest_table(backtest_results)
    print("\n" + str(table))

    # Save results
    output_path = Path('data/processed/backtest_results_regime.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(backtest_results, f)

    logger.success(f"Results saved to {output_path}")

    # Summary by ticker
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY BY TICKER")
    logger.info("=" * 80)

    for ticker in backtest_results['tickers']:
        logger.info(f"\n{ticker}:")
        ticker_results = [
            r for r in backtest_results['results']
            if r['ticker'] == ticker
        ]

        for result in ticker_results:
            model_name = result['model_name']
            trading = result['trading']

            logger.info(f"  {model_name}:")
            logger.info(f"    Total return:  {trading['total_return']:.2%}")
            logger.info(f"    Sharpe ratio:  {trading['sharpe']:.2f}")
            logger.info(f"    Max drawdown:  {trading['max_drawdown']:.2%}")
            logger.info(f"    Calmar ratio:  {trading['calmar_ratio']:.2f}")
            logger.info(f"    Win rate:      {trading['win_rate']:.2%}")

            if result['classification']:
                logger.info(
                    f"    Accuracy:      "
                    f"{result['classification']['accuracy']:.3f}"
                )
                logger.info(
                    f"    ROC-AUC:       "
                    f"{result['classification']['roc_auc']:.3f}"
                )

    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKTEST WITH REGIME COMPLETE")
    logger.info("=" * 80)
    logger.info(f"OOS period:       {oos_start} to {oos_end}")
    logger.info(f"Regime features:  {len(REGIME_COLS)} columns")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()