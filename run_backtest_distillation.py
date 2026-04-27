#!/usr/bin/env python3
"""
Backtest Distilled EBM vs LightGBM Teacher on OOS Period (2020-2024)

Evaluates the distilled EBM model (student) trained with knowledge distillation
against the LightGBM teacher model on the out-of-sample period.

Uses the best temperature T selected during walk-forward validation.
Implements both long-short and long-only strategies.
"""

import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.ingestion.loader import DataLoader
from src.models import WalkForwardCV

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def calculate_metrics_long_short(predictions, returns, proba_positive, hard_labels):
    """
    Calculate metrics for long-short strategy.
    
    Args:
        predictions: Predictions in {-1, 1}
        returns: Forward returns (ret_1d_forward)
        proba_positive: Probability of positive class
        hard_labels: True labels in {0, 1}
        
    Returns:
        Dictionary with metrics
    """
    # Strategy returns: long when predict 1, short when predict -1
    strategy_returns = predictions * returns
    
    # Cumulative return
    total_return = (1 + strategy_returns).prod() - 1
    
    # Sharpe ratio (annualized)
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Position counts
    n_long = (predictions == 1).sum()
    n_short = (predictions == -1).sum()
    
    # Classification metrics
    accuracy = accuracy_score(hard_labels, (predictions == 1).astype(int))
    roc_auc = roc_auc_score(hard_labels, proba_positive)
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'n_long': n_long,
        'n_short': n_short,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }


def calculate_metrics_long_only(predictions, returns, proba_positive, hard_labels):
    """
    Calculate metrics for long-only strategy.
    
    Args:
        predictions: Predictions in {-1, 1}
        returns: Forward returns (ret_1d_forward)
        proba_positive: Probability of positive class
        hard_labels: True labels in {0, 1}
        
    Returns:
        Dictionary with metrics
    """
    # Position: 1 when predict 1, 0 when predict -1
    position = (predictions == 1).astype(int)
    
    # Strategy returns: only when long
    strategy_returns = position * returns
    
    # Cumulative return
    total_return = (1 + strategy_returns).prod() - 1
    
    # Sharpe ratio (annualized, computed over all days)
    if strategy_returns.std() > 0:
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate: fraction of long trades with positive return
    active_returns = strategy_returns[position == 1]
    if len(active_returns) > 0:
        win_rate = (active_returns > 0).sum() / len(active_returns)
    else:
        win_rate = 0
    
    # Position count
    n_long = position.sum()
    
    # Classification metrics
    accuracy = accuracy_score(hard_labels, (predictions == 1).astype(int))
    roc_auc = roc_auc_score(hard_labels, proba_positive)
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_long': n_long,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }


def calculate_benchmark_metrics(returns, strategy='long_short'):
    """
    Calculate buy & hold benchmark metrics.
    
    Args:
        returns: Forward returns (ret_1d_forward)
        strategy: 'long_short' or 'long_only'
        
    Returns:
        Dictionary with metrics
    """
    # Buy & hold: always long
    strategy_returns = returns
    
    # Cumulative return
    total_return = (1 + strategy_returns).prod() - 1
    
    # Sharpe ratio (annualized)
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    if strategy == 'long_only':
        win_rate = (returns > 0).sum() / len(returns)
        n_long = len(returns)
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_long': n_long
        }
    else:
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }


def get_predictions_and_proba(model, X_oos, original_labels):
    """
    Get predictions and probabilities, handling different class encodings.
    
    Args:
        model: Trained model
        X_oos: OOS features
        original_labels: Original labels in {-1, 1}
        
    Returns:
        Tuple of (predictions in {-1, 1}, proba_positive, hard_labels in {0, 1})
    """
    # Get model predictions
    pred = model.predict(X_oos)
    proba = model.predict_proba(X_oos)
    
    # Check model classes
    classes = model.classes_
    
    if np.array_equal(classes, [-1, 1]):
        # Model trained on {-1, 1}
        predictions = pred  # Already in {-1, 1}
        proba_positive = proba[:, 1]  # Probability of class 1
    elif np.array_equal(classes, [0, 1]):
        # Model trained on {0, 1}, need to map back
        predictions = np.where(pred == 1, 1, -1)  # Map 0->-1, 1->1
        proba_positive = proba[:, 1]  # Probability of class 1
    else:
        raise ValueError(f"Unexpected model classes: {classes}")
    
    # Convert original labels to {0, 1} for classification metrics
    hard_labels = (original_labels == 1).astype(int)
    
    return predictions, proba_positive, hard_labels


def main():
    """Run backtest for distilled models."""
    logger.add(
        "logs/backtest_distillation_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    logger.info("=" * 80)
    logger.info("BACKTEST: DISTILLED EBM VS LIGHTGBM TEACHER")
    logger.info("=" * 80)
    logger.info("OOS Period: 2020-2024")
    logger.info("=" * 80)
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load distillation results
    logger.info("\nLoading distillation results...")
    distill_path = Path('data/processed/walk_forward_distillation_results.pkl')
    
    if not distill_path.exists():
        raise FileNotFoundError(
            f"Distillation results not found: {distill_path}\n"
            "Please run run_walk_forward_distillation.py first."
        )
    
    with open(distill_path, 'rb') as f:
        distill_results = pickle.load(f)
    
    best_T = distill_results['best_T']
    logger.info(f"Best temperature: T={best_T}")
    
    # Extract fold 9 (last fold)
    fold_9 = distill_results['all_fold_results'][-1]
    logger.info(f"Using fold {fold_9['fold_number']} models")
    
    # Extract models
    lightgbm_model = fold_9['lightgbm_model']
    ebm_distilled_model = fold_9['ebm_distilled_models'][best_T]
    
    logger.info("Models loaded:")
    logger.info(f"  - LightGBM teacher")
    logger.info(f"  - EBM distilled (T={best_T})")
    
    # Load OOS data
    logger.info("\nLoading OOS data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()
    
    # Filter to OOS period
    oos_start = '2020-01-01'
    oos_end = '2024-12-31'
    
    oos_data = data[
        (data.index.get_level_values('date') >= oos_start) &
        (data.index.get_level_values('date') <= oos_end)
    ].copy()
    
    logger.info(f"OOS data: {oos_data.shape}")
    logger.info(f"Date range: {oos_data.index.get_level_values('date').min()} to "
               f"{oos_data.index.get_level_values('date').max()}")
    
    # Get feature columns
    wf_cv = WalkForwardCV(config)
    feature_names = wf_cv.get_feature_names(oos_data)
    logger.info(f"Features: {len(feature_names)}")
    
    # Drop last row per ticker (NaN ret_1d_forward)
    original_len = len(oos_data)
    oos_data = oos_data.groupby(level='ticker', group_keys=False).apply(
        lambda x: x.iloc[:-1]
    )
    logger.info(f"Dropped last row per ticker: {original_len} -> {len(oos_data)} samples")
    
    # Prepare features and labels
    X_oos = oos_data[feature_names].values
    y_oos = oos_data['label'].values  # Original labels in {-1, 1}
    returns = oos_data['ret_1d_forward'].values
    
    logger.info(f"Final OOS samples: {len(X_oos)}")
    logger.info(f"Label distribution: {np.bincount(y_oos.astype(int) + 1)}")  # Map to {0, 1, 2}
    
    # Get predictions for both models
    logger.info("\n--- Generating Predictions ---")
    
    logger.info("LightGBM teacher...")
    lgbm_pred, lgbm_proba, hard_labels = get_predictions_and_proba(
        lightgbm_model, X_oos, y_oos
    )
    
    logger.info("EBM distilled...")
    ebm_pred, ebm_proba, _ = get_predictions_and_proba(
        ebm_distilled_model, X_oos, y_oos
    )
    
    # Calculate metrics for long-short strategy
    logger.info("\n--- Long-Short Strategy ---")
    
    lgbm_ls = calculate_metrics_long_short(lgbm_pred, returns, lgbm_proba, hard_labels)
    ebm_ls = calculate_metrics_long_short(ebm_pred, returns, ebm_proba, hard_labels)
    bench_ls = calculate_benchmark_metrics(returns, strategy='long_short')
    
    # Calculate metrics for long-only strategy
    logger.info("\n--- Long-Only Strategy ---")
    
    lgbm_lo = calculate_metrics_long_only(lgbm_pred, returns, lgbm_proba, hard_labels)
    ebm_lo = calculate_metrics_long_only(ebm_pred, returns, ebm_proba, hard_labels)
    bench_lo = calculate_benchmark_metrics(returns, strategy='long_only')
    
    # Print results
    print("\n" + "=" * 80)
    print("LONG-SHORT RESULTS")
    print("=" * 80)
    print(f"{'Model':<25} | {'Return':>8} | {'Sharpe':>6} | {'Max DD':>8} | {'Acc':>6} | {'AUC':>6}")
    print("-" * 80)
    print(f"{'LightGBM (teacher)':<25} | {lgbm_ls['total_return']:>7.1%} | "
          f"{lgbm_ls['sharpe']:>6.2f} | {lgbm_ls['max_drawdown']:>7.1%} | "
          f"{lgbm_ls['accuracy']:>6.3f} | {lgbm_ls['roc_auc']:>6.3f}")
    print(f"{'EBM distilled (T=' + str(best_T) + ')':<25} | {ebm_ls['total_return']:>7.1%} | "
          f"{ebm_ls['sharpe']:>6.2f} | {ebm_ls['max_drawdown']:>7.1%} | "
          f"{ebm_ls['accuracy']:>6.3f} | {ebm_ls['roc_auc']:>6.3f}")
    print(f"{'Buy & Hold':<25} | {bench_ls['total_return']:>7.1%} | "
          f"{bench_ls['sharpe']:>6.2f} | {bench_ls['max_drawdown']:>7.1%} | "
          f"{'—':>6} | {'—':>6}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("LONG-ONLY RESULTS")
    print("=" * 80)
    print(f"{'Model':<25} | {'Return':>8} | {'Sharpe':>6} | {'Max DD':>8} | {'Win Rate':>9} | {'N Long':>7}")
    print("-" * 80)
    print(f"{'LightGBM (teacher)':<25} | {lgbm_lo['total_return']:>7.1%} | "
          f"{lgbm_lo['sharpe']:>6.2f} | {lgbm_lo['max_drawdown']:>7.1%} | "
          f"{lgbm_lo['win_rate']:>8.1%} | {lgbm_lo['n_long']:>7}")
    print(f"{'EBM distilled (T=' + str(best_T) + ')':<25} | {ebm_lo['total_return']:>7.1%} | "
          f"{ebm_lo['sharpe']:>6.2f} | {ebm_lo['max_drawdown']:>7.1%} | "
          f"{ebm_lo['win_rate']:>8.1%} | {ebm_lo['n_long']:>7}")
    print(f"{'Buy & Hold':<25} | {bench_lo['total_return']:>7.1%} | "
          f"{bench_lo['sharpe']:>6.2f} | {bench_lo['max_drawdown']:>7.1%} | "
          f"{bench_lo['win_rate']:>8.1%} | {bench_lo['n_long']:>7}")
    print("=" * 80)
    
    # Save results
    logger.info("\n--- Saving Results ---")
    
    results = {
        'best_T': best_T,
        'oos_start': oos_start,
        'oos_end': oos_end,
        'long_short': {
            'lightgbm': lgbm_ls,
            'ebm_distilled': ebm_ls,
            'benchmark': bench_ls
        },
        'long_only': {
            'lightgbm': lgbm_lo,
            'ebm_distilled': ebm_lo,
            'benchmark': bench_lo
        }
    }
    
    output_path = Path('data/processed/backtest_distillation_results.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.success(f"Results saved to {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
