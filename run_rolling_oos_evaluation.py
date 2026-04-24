#!/usr/bin/env python3
"""
Rolling Walk-Forward Evaluation on OOS Period (2020-2024)

This script implements a rolling walk-forward approach where the model is
retrained each year on expanding data:
- Fold 1: Train 2008-2020, predict 2021
- Fold 2: Train 2008-2021, predict 2022
- Fold 3: Train 2008-2022, predict 2023
- Fold 4: Train 2008-2023, predict 2024

This provides a more realistic evaluation than using a single model trained
on 2008-2019 data, as it allows the model to adapt to changing market conditions.
"""

import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime

from src.ingestion.loader import DataLoader
from src.models.train import ModelTrainer
from src.models.regime_detection import RegimeDetector

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Define the 16 pure technical features (baseline iteration 1)
TECHNICAL_FEATURES = [
    # Returns (3)
    'ret_1d', 'ret_5d', 'ret_21d',
    # Volatility (2)
    'vol_20d', 'atr_14',
    # Technical indicators (10)
    'rsi_14', 'macd_line', 'macd_signal', 'macd_hist',
    'bb_mid', 'bb_upper', 'bb_lower', 'bb_pct', 'bb_width',
    'volume_ratio',
    # Trend context (1)
    'sma_200_dist'
]


def create_rolling_oos_folds(data, config):
    """
    Create rolling OOS folds for 2020-2024.
    
    Returns list of dicts with train/test splits for each year.
    """
    folds = []
    
    # Fold 1: Train 2008-2020, Test 2021
    folds.append({
        'fold_id': 1,
        'train_start': '2008-01-01',
        'train_end': '2020-12-31',
        'test_start': '2021-01-01',
        'test_end': '2021-12-31',
        'test_year': 2021
    })
    
    # Fold 2: Train 2008-2021, Test 2022
    folds.append({
        'fold_id': 2,
        'train_start': '2008-01-01',
        'train_end': '2021-12-31',
        'test_start': '2022-01-01',
        'test_end': '2022-12-31',
        'test_year': 2022
    })
    
    # Fold 3: Train 2008-2022, Test 2023
    folds.append({
        'fold_id': 3,
        'train_start': '2008-01-01',
        'train_end': '2022-12-31',
        'test_start': '2023-01-01',
        'test_end': '2023-12-31',
        'test_year': 2023
    })
    
    # Fold 4: Train 2008-2023, Test 2024
    folds.append({
        'fold_id': 4,
        'train_start': '2008-01-01',
        'train_end': '2023-12-31',
        'test_start': '2024-01-01',
        'test_end': '2024-12-31',
        'test_year': 2024
    })
    
    return folds


def get_feature_columns(data_df, iteration):
    """
    Get feature columns for the given iteration.
    
    Uses TECHNICAL_FEATURES list to ensure we use exactly the same 17 features
    as in walk_forward.py, plus ticker_id.
    
    Args:
        data_df: DataFrame to get columns from
        iteration: 1 for baseline, 2 for with regime features
        
    Returns:
        List of feature column names
    """
    # Start with technical features that exist in the data
    available_cols = data_df.columns.tolist()
    feature_cols = [col for col in TECHNICAL_FEATURES if col in available_cols]
    
    # Add ticker_id
    feature_cols.append('ticker_id')
    
    # For iteration 2, add regime features
    if iteration == 2:
        regime_features = ['regime_state', 'regime_prob_0', 'regime_prob_1', 'regime_prob_2']
        for col in regime_features:
            if col in available_cols:
                feature_cols.append(col)
    
    return feature_cols


def train_fold(data, fold_info, config, iteration):
    """Train models for a single fold."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FOLD {fold_info['fold_id']}: TRAIN {fold_info['train_start']} to {fold_info['train_end']}, TEST {fold_info['test_year']}")
    logger.info(f"{'=' * 80}")
    
    # Filter data for this fold
    train_data = data[
        (data.index.get_level_values('date') >= fold_info['train_start']) &
        (data.index.get_level_values('date') <= fold_info['train_end'])
    ].copy()
    
    test_data = data[
        (data.index.get_level_values('date') >= fold_info['test_start']) &
        (data.index.get_level_values('date') <= fold_info['test_end'])
    ].copy()
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Add ticker_id to both train and test data
    train_ticker_values = train_data.index.get_level_values('ticker')
    train_data['ticker_id'] = (train_ticker_values == 'USO').astype(int)
    
    test_ticker_values = test_data.index.get_level_values('ticker')
    test_data['ticker_id'] = (test_ticker_values == 'USO').astype(int)
    
    # Handle regime features for iteration 2
    if iteration == 2:
        logger.info("Detecting regimes on training data...")
        regime_detector = RegimeDetector(config)
        
        # Fit GMM on training data
        regime_detector.fit(train_data)
        
        # Get regime labels and probabilities for training data
        train_regime_labels, train_regime_probs = regime_detector.predict(train_data)
        
        # Add regime features to training data
        train_data['regime_state'] = train_regime_labels
        train_data['regime_prob_0'] = train_regime_probs[:, 0]
        train_data['regime_prob_1'] = train_regime_probs[:, 1]
        train_data['regime_prob_2'] = train_regime_probs[:, 2]
        
        # Predict regimes for test data
        test_regime_labels, test_regime_probs = regime_detector.predict(test_data)
        
        test_data['regime_state'] = test_regime_labels
        test_data['regime_prob_0'] = test_regime_probs[:, 0]
        test_data['regime_prob_1'] = test_regime_probs[:, 1]
        test_data['regime_prob_2'] = test_regime_probs[:, 2]
    
    # Get feature columns using the fixed function
    feature_cols = get_feature_columns(train_data, iteration)
    
    logger.info(f"Features selected: {len(feature_cols)}")
    logger.info(f"Feature list: {feature_cols}")
    
    # ASSERTION: Verify feature count
    expected_count = 17 if iteration == 1 else 21  # 16 technical + ticker_id, or 16 + ticker_id + 4 regime
    if len(feature_cols) != expected_count:
        error_msg = f"Feature count mismatch! Expected {expected_count}, got {len(feature_cols)}"
        logger.error(error_msg)
        logger.error(f"Features: {feature_cols}")
        
        # Check which columns are extra or missing
        if iteration == 1:
            expected_features = TECHNICAL_FEATURES + ['ticker_id']
        else:
            expected_features = TECHNICAL_FEATURES + ['ticker_id', 'regime_state', 'regime_prob_0', 'regime_prob_1', 'regime_prob_2']
        
        extra = set(feature_cols) - set(expected_features)
        missing = set(expected_features) - set(feature_cols)
        
        if extra:
            logger.error(f"Extra columns: {extra}")
        if missing:
            logger.error(f"Missing columns: {missing}")
        
        raise ValueError(error_msg)
    
    logger.success(f"✓ Feature count verified: {len(feature_cols)} features")
    
    # Prepare data for training
    X_train = train_data[feature_cols].values
    y_train = train_data['label'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['label'].values
    
    # Convert labels from {-1, 0, 1} to {0, 1}
    # label == 1 (take profit) -> 1
    # label == -1 or 0 (stop loss or time barrier) -> 0
    y_train_binary = (y_train == 1).astype(int)
    y_test_binary = (y_test == 1).astype(int)
    
    logger.info(f"Train label distribution: {np.bincount(y_train_binary)}")
    logger.info(f"Test label distribution: {np.bincount(y_test_binary)}")
    
    # Train models
    trainer = ModelTrainer(config)
    
    # Train LightGBM
    logger.info("Training LightGBM...")
    lgbm_model = trainer._train_lightgbm(X_train, y_train_binary, fold_info['fold_id'])
    
    # Train EBM
    logger.info("Training EBM...")
    ebm_model = trainer._train_ebm(X_train, y_train_binary, fold_info['fold_id'])
    
    # Evaluate on test set
    results = {}
    
    for ticker in ['SPY', 'USO']:
        ticker_mask = test_data.index.get_level_values('ticker') == ticker
        X_test_ticker = X_test[ticker_mask]
        y_test_ticker = y_test_binary[ticker_mask]
        
        # Get predictions
        lgbm_pred_proba = lgbm_model.predict_proba(X_test_ticker)[:, 1]
        lgbm_pred = (lgbm_pred_proba >= 0.5).astype(int)
        
        ebm_pred_proba = ebm_model.predict_proba(X_test_ticker)[:, 1]
        ebm_pred = (ebm_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        results[ticker] = {
            'lightgbm': {
                'accuracy': accuracy_score(y_test_ticker, lgbm_pred),
                'roc_auc': roc_auc_score(y_test_ticker, lgbm_pred_proba),
                'predictions': lgbm_pred,
                'pred_proba': lgbm_pred_proba
            },
            'ebm': {
                'accuracy': accuracy_score(y_test_ticker, ebm_pred),
                'roc_auc': roc_auc_score(y_test_ticker, ebm_pred_proba),
                'predictions': ebm_pred,
                'pred_proba': ebm_pred_proba
            },
            'y_true': y_test_ticker,
            'test_data': test_data[ticker_mask]
        }
        
        logger.info(f"{ticker} - LightGBM: Acc={results[ticker]['lightgbm']['accuracy']:.3f}, AUC={results[ticker]['lightgbm']['roc_auc']:.3f}")
        logger.info(f"{ticker} - EBM: Acc={results[ticker]['ebm']['accuracy']:.3f}, AUC={results[ticker]['ebm']['roc_auc']:.3f}")
    
    return results


def calculate_trading_metrics(predictions, test_data, strategy='long_only'):
    """Calculate trading metrics from predictions."""
    # Get returns (use ret_1d_forward to avoid look-ahead bias)
    returns = test_data['ret_1d_forward'].values
    
    # Drop NaN values (last row per ticker)
    nan_mask = ~np.isnan(returns)
    returns = returns[nan_mask]
    predictions = predictions[nan_mask]
    
    # Convert predictions to signals
    if strategy == 'long_only':
        # Long when predict 1, cash when predict 0
        signals = predictions.copy()
    else:  # long_short
        # Long when predict 1, short when predict 0
        signals = predictions.copy()
        signals[signals == 0] = -1
    
    # Calculate strategy returns
    strategy_returns = signals * returns
    
    # Calculate cumulative return
    cumulative_return = (1 + strategy_returns).prod() - 1
    
    # Calculate Sharpe ratio (annualized, assuming daily returns)
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': cumulative_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }


def main():
    """Run rolling OOS evaluation."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"logs/rolling_oos_fixed_{timestamp}.log"
    
    logger.add(
        log_file,
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    logger.info("=" * 80)
    logger.info("ROLLING WALK-FORWARD EVALUATION: OOS PERIOD 2020-2024")
    logger.info("=" * 80)
    logger.info("Retraining models each year on expanding window")
    logger.info("=" * 80)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info("\nLoading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()
    
    logger.info(f"Data loaded: {data.shape}")
    logger.info(f"Date range: {data.index.get_level_values('date').min()} to {data.index.get_level_values('date').max()}")
    
    # Create folds
    folds = create_rolling_oos_folds(data, config)
    
    # Run both iterations
    all_results = {
        'iteration_1': {},  # Baseline
        'iteration_2': {}   # With regime features
    }
    
    for iteration in [1, 2]:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ITERATION {iteration}: {'BASELINE (16 pure technical features + ticker_id)' if iteration == 1 else 'WITH REGIME FEATURES (16 technical + ticker_id + 4 regime)'}")
        logger.info(f"{'=' * 80}")
        
        iteration_results = []
        
        for fold_info in folds:
            fold_results = train_fold(data, fold_info, config, iteration)
            fold_results['fold_info'] = fold_info
            iteration_results.append(fold_results)
        
        # Aggregate results across all folds
        aggregated = {
            'SPY': {'lightgbm': {}, 'ebm': {}},
            'USO': {'lightgbm': {}, 'ebm': {}}
        }
        
        for ticker in ['SPY', 'USO']:
            for model in ['lightgbm', 'ebm']:
                # Collect all predictions and true labels
                all_preds = []
                all_proba = []
                all_true = []
                all_test_data = []
                
                for fold_results in iteration_results:
                    all_preds.append(fold_results[ticker][model]['predictions'])
                    all_proba.append(fold_results[ticker][model]['pred_proba'])
                    all_true.append(fold_results[ticker]['y_true'])
                    all_test_data.append(fold_results[ticker]['test_data'])
                
                # Concatenate
                all_preds = np.concatenate(all_preds)
                all_proba = np.concatenate(all_proba)
                all_true = np.concatenate(all_true)
                all_test_data = pd.concat(all_test_data)
                
                # Calculate overall metrics
                from sklearn.metrics import accuracy_score, roc_auc_score
                
                aggregated[ticker][model]['accuracy'] = accuracy_score(all_true, all_preds)
                aggregated[ticker][model]['roc_auc'] = roc_auc_score(all_true, all_proba)
                
                # Calculate trading metrics
                trading_metrics = calculate_trading_metrics(all_preds, all_test_data, strategy='long_only')
                aggregated[ticker][model]['trading'] = trading_metrics
        
        all_results[f'iteration_{iteration}'] = {
            'fold_results': iteration_results,
            'aggregated': aggregated
        }
    
    # Save results
    output_dir = Path('data/processed/rolling_oos')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'rolling_oos_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    logger.success(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ROLLING OOS RESULTS SUMMARY (2020-2024)")
    print("=" * 80)
    
    for iteration in [1, 2]:
        print(f"\n{'ITERATION ' + str(iteration)}: {'BASELINE' if iteration == 1 else 'WITH REGIME FEATURES'}")
        print("-" * 80)
        
        agg = all_results[f'iteration_{iteration}']['aggregated']
        
        for ticker in ['SPY', 'USO']:
            print(f"\n{ticker}:")
            for model in ['lightgbm', 'ebm']:
                metrics = agg[ticker][model]
                print(f"  {model.upper()}:")
                print(f"    Accuracy: {metrics['accuracy']:.3f}")
                print(f"    ROC-AUC:  {metrics['roc_auc']:.3f}")
                print(f"    Return:   {metrics['trading']['total_return']:>8.2%}")
                print(f"    Sharpe:   {metrics['trading']['sharpe']:>6.2f}")
                print(f"    MaxDD:    {metrics['trading']['max_drawdown']:>8.2%}")
    
    # Compare with fixed-model results
    print("\n" + "=" * 80)
    print("COMPARISON: ROLLING vs FIXED-MODEL OOS")
    print("=" * 80)
    
    try:
        # Load single-model results for comparison
        with open('data/processed/backtest_results_longonly.pkl', 'rb') as f:
            fixed_results = pickle.load(f)
        
        print("\nBaseline - Long-Only Strategy:")
        print("-" * 80)
        
        for ticker in ['SPY', 'USO']:
            print(f"\n{ticker}:")
            
            # Rolling results
            rolling_lgbm = all_results['iteration_1']['aggregated'][ticker]['lightgbm']
            rolling_ebm = all_results['iteration_1']['aggregated'][ticker]['ebm']
            
            # Single-model results
            single_lgbm = next(r for r in fixed_results['iteration_1'] 
                            if r['ticker'] == ticker and r['model_name'] == 'lightgbm')
            single_ebm = next(r for r in fixed_results['iteration_1'] 
                           if r['ticker'] == ticker and r['model_name'] == 'ebm')
            
            print(f"  LightGBM:")
            print(f"    Rolling:      Return {rolling_lgbm['trading']['total_return']:>8.2%}, Sharpe {rolling_lgbm['trading']['sharpe']:>6.2f}")
            print(f"    Single-model: Return {single_lgbm['metrics']['total_return']:>8.2%}, Sharpe {single_lgbm['metrics']['sharpe']:>6.2f}")
            
            print(f"  EBM:")
            print(f"    Rolling:      Return {rolling_ebm['trading']['total_return']:>8.2%}, Sharpe {rolling_ebm['trading']['sharpe']:>6.2f}")
            print(f"    Single-model: Return {single_ebm['metrics']['total_return']:>8.2%}, Sharpe {single_ebm['metrics']['sharpe']:>6.2f}")
        
        logger.info("\nComparison complete")
        
    except FileNotFoundError:
        logger.warning("Could not load single-model results for comparison")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("ROLLING OOS EVALUATION COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
