"""
Backtesting without Regime Detection (Iteration 1 - Baseline)

Runs backtesting on the OOS period 2020-2024 using models trained WITHOUT
regime features. This is the baseline iteration for comparison with the
regime-enhanced iteration 2.

Uses models from fold 8 (trained on 2008-2019) to predict on OOS period.
No look-ahead bias is introduced.

Results are saved to data/processed/backtest_results.pkl.
"""

import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger

from src.ingestion.loader import DataLoader
from src.models.backtest import Backtester


def main():
    """Run backtesting without regime features on OOS period."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    logger.add(
        "logs/backtest_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("=" * 80)
    logger.info("BACKTESTING - BASELINE (NO REGIME FEATURES)")
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

    # Load walk-forward results (baseline, no regime)
    wf_path = Path('data/processed/walk_forward_results.pkl')
    if not wf_path.exists():
        logger.error(f"Walk-forward results not found at {wf_path}")
        logger.error(
            "Please run walk-forward first: python run_walk_forward.py"
        )
        return

    logger.info(f"Loading walk-forward results from {wf_path}...")
    with open(wf_path, 'rb') as f:
        wf_results = pickle.load(f)

    all_fold_results = wf_results['all_fold_results']
    logger.info(f"Loaded {len(all_fold_results)} fold results")

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
    logger.info("FILTERING OOS PERIOD")
    logger.info("=" * 80)
    logger.info(f"OOS period: {oos_start} to {oos_end}")

    dates = data.index.get_level_values('date')
    mask = (dates >= oos_start) & (dates <= oos_end)
    data_oos = data[mask].copy()

    logger.info(f"OOS data: {data_oos.shape}")
    logger.info(f"OOS features: {data_oos.shape[1]}")

    # Run backtest
    logger.info("")
    logger.info("=" * 80)
    logger.info("RUNNING BACKTEST")
    logger.info("=" * 80)

    backtester = Backtester(config)
    backtest_results = backtester.run_backtest(all_fold_results, data_oos)

    # Results table
    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS TABLE")
    logger.info("=" * 80)

    table = backtester.get_backtest_table(backtest_results)
    print("\n" + str(table))

    # Save results
    output_path = Path('data/processed/backtest_results.pkl')
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
    logger.info("BACKTEST BASELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"OOS period:       {oos_start} to {oos_end}")
    logger.info(f"Regime features:  NO (baseline)")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
