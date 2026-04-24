#!/usr/bin/env python3
"""
Run long-only backtest for both iterations using existing trained models.
Long-only strategy: Long when predict +1, Cash (return=0) when predict -1.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'backtest_longonly_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_long_only_metrics(predictions, returns, labels):
    """
    Calculate long-only trading metrics.
    
    Long when predict +1, cash (return=0) when predict -1.
    """
    # Convert predictions to positions: 1 for long, 0 for cash
    positions = (predictions == 1).astype(int)
    
    # Strategy returns: position * market_return
    strategy_returns = positions * returns
    
    # Cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    # Sharpe ratio (annualized, 252 trading days)
    if strategy_returns.std() > 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Maximum drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate (only for long positions)
    long_returns = strategy_returns[positions == 1]
    if len(long_returns) > 0:
        win_rate = (long_returns > 0).sum() / len(long_returns)
    else:
        win_rate = 0.0
    
    # Number of trades
    n_long = positions.sum()
    n_short = 0  # No shorts in long-only
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_long': n_long,
        'n_short': n_short,
        'cumulative_returns': cumulative_returns
    }

def run_long_only_backtest(iteration_name, results_file):
    """Run long-only backtest for a given iteration."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running long-only backtest for {iteration_name}")
    logger.info(f"{'='*80}")
    
    # Load existing results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded results from {results_file}")
    logger.info(f"OOS period: {results['oos_start']} to {results['oos_end']}")
    
    # Load price data for returns
    with pd.HDFStore('data/processed/assets.h5', 'r') as store:
        # Check available keys
        keys = store.keys()
        logger.info(f"Available keys in HDF5: {keys}")
        
        # Use the correct key (likely 'df' or similar)
        if '/df' in keys:
            df = store['df']
        elif 'df' in keys:
            df = store['df']
        else:
            # Use first available key
            df = store[keys[0]]
    
    # Filter to OOS period
    oos_start = pd.Timestamp(results['oos_start'])
    oos_end = pd.Timestamp(results['oos_end'])
    df_oos = df.loc[(df.index.get_level_values('date') >= oos_start) & 
                    (df.index.get_level_values('date') <= oos_end)]
    
    long_only_results = []
    
    # Process each model/ticker combination
    for item in results['results']:
        model_name = item['model_name']
        ticker = item['ticker']
        
        logger.info(f"\nProcessing {model_name.upper()} - {ticker}")
        
        # Get predictions and returns for this ticker
        ticker_data = df_oos.loc[ticker]
        predictions = item['predictions']
        
        # Get returns (next day return)
        returns = ticker_data['ret_1d'].values[:len(predictions)]
        labels = ticker_data['label'].values[:len(predictions)]
        
        # Calculate long-only metrics
        metrics = calculate_long_only_metrics(
            predictions=predictions,
            returns=pd.Series(returns),
            labels=labels
        )
        
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Long Trades: {metrics['n_long']}")
        
        long_only_results.append({
            'model_name': model_name,
            'ticker': ticker,
            'metrics': metrics
        })
    
    return long_only_results

def main():
    logger.info("="*80)
    logger.info("LONG-ONLY BACKTEST - BOTH ITERATIONS")
    logger.info("="*80)
    
    # Run for both iterations
    iter1_results = run_long_only_backtest(
        "Iteration 1 (Baseline)",
        "data/processed/backtest_results.pkl"
    )
    
    iter2_results = run_long_only_backtest(
        "Iteration 2 (Regime)",
        "data/processed/backtest_results_regime.pkl"
    )
    
    # Save results
    output = {
        'iteration_1': iter1_results,
        'iteration_2': iter2_results,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = Path('data/processed/backtest_results_longonly_corrected.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"{'='*80}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - ITERATION 1 (BASELINE)")
    logger.info("="*80)
    for item in iter1_results:
        logger.info(f"\n{item['model_name'].upper()} - {item['ticker']}:")
        logger.info(f"  Return: {item['metrics']['total_return']:>8.2%}")
        logger.info(f"  Sharpe: {item['metrics']['sharpe']:>8.2f}")
        logger.info(f"  MaxDD:  {item['metrics']['max_drawdown']:>8.2%}")
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY - ITERATION 2 (REGIME)")
    logger.info("="*80)
    for item in iter2_results:
        logger.info(f"\n{item['model_name'].upper()} - {item['ticker']}:")
        logger.info(f"  Return: {item['metrics']['total_return']:>8.2%}")
        logger.info(f"  Sharpe: {item['metrics']['sharpe']:>8.2f}")
        logger.info(f"  MaxDD:  {item['metrics']['max_drawdown']:>8.2%}")

if __name__ == '__main__':
    main()
