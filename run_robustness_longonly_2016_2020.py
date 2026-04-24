#!/usr/bin/env python3
"""
Robustness Check: Long-Only Strategy on Alternative OOS Window (2016-2020)

Trains models on 2008-2016 and tests on 2016-2020 using long-only strategy.
This is a standalone robustness check, separate from the main thesis results.

Training period: 2008-01-01 to 2016-01-01
Test period: 2016-01-01 to 2020-01-01

Uses same 23 baseline features and hyperparameters as main thesis.
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
from src.models.walk_forward import WalkForwardCV
from src.models.train import ModelTrainer
from src.models.backtest import Backtester

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def main():
    """Run robustness check: train 2008-2016, test 2016-2020, long-only."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"logs/robustness_longonly_2016_2020_{timestamp}.log"
    
    logger.add(
        log_file,
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    logger.info("=" * 80)
    logger.info("ROBUSTNESS CHECK: LONG-ONLY STRATEGY 2016-2020")
    logger.info("=" * 80)
    logger.info("Training period: 2008-2016")
    logger.info("Test period: 2016-2020")
    logger.info("Strategy: Long-only (long when predict +1, cash when predict -1)")
    logger.info("=" * 80)
    
    # Load config and modify dates for this robustness check
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override walk-forward dates for 2008-2016 training
    config['models']['walk_forward']['train_start'] = '2008-01-01'
    config['models']['walk_forward']['train_end'] = '2016-01-01'
    config['models']['walk_forward']['test_start'] = '2016-01-01'
    config['models']['walk_forward']['test_end'] = '2020-01-01'
    
    logger.info("Modified config for robustness check:")
    logger.info(f"  Train: {config['models']['walk_forward']['train_start']} to {config['models']['walk_forward']['train_end']}")
    logger.info(f"  Test:  {config['models']['walk_forward']['test_start']} to {config['models']['walk_forward']['test_end']}")
    
    # Load data
    logger.info("\nLoading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()
    
    logger.info(f"Data loaded: {data.shape}")
    logger.info(
        f"Date range: {data.index.get_level_values('date').min()} to "
        f"{data.index.get_level_values('date').max()}"
    )
    
    # Train models using walk-forward on 2008-2016
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING MODELS (2008-2016)")
    logger.info("=" * 80)
    
    trainer = ModelTrainer(config)
    all_fold_results = trainer.train_all_folds(data)
    
    logger.info(f"\nTrained {len(all_fold_results)} folds")
    
    # Run backtest on 2016-2020 with LONG-SHORT strategy
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST: LONG-SHORT STRATEGY (2016-2020)")
    logger.info("=" * 80)
    
    backtester = Backtester(config)
    backtest_longshort = backtester.run_backtest(
        all_fold_results, 
        data, 
        strategy='long_short'
    )
    
    # Run backtest on 2016-2020 with LONG-ONLY strategy
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST: LONG-ONLY STRATEGY (2016-2020)")
    logger.info("=" * 80)
    
    backtest_longonly = backtester.run_backtest(
        all_fold_results,
        data,
        strategy='long_only'
    )
    
    # Create results table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS TABLE: 2016-2020 ROBUSTNESS CHECK")
    logger.info("=" * 80)
    
    results_data = []
    
    for ticker in ['SPY', 'USO']:
        # Long-short results
        ls_results = [r for r in backtest_longshort['results'] if r['ticker'] == ticker]
        # Long-only results
        lo_results = [r for r in backtest_longonly['results'] if r['ticker'] == ticker]
        
        for model_name in ['lightgbm', 'ebm', 'benchmark']:
            ls_result = next((r for r in ls_results if r['model_name'] == model_name), None)
            lo_result = next((r for r in lo_results if r['model_name'] == model_name), None)
            
            if ls_result and lo_result:
                results_data.append({
                    'Ticker': ticker,
                    'Model': model_name.upper(),
                    'LS_Return': ls_result['trading']['total_return'],
                    'LS_Sharpe': ls_result['trading']['sharpe'],
                    'LS_MaxDD': ls_result['trading']['max_drawdown'],
                    'LO_Return': lo_result['trading']['total_return'],
                    'LO_Sharpe': lo_result['trading']['sharpe'],
                    'LO_MaxDD': lo_result['trading']['max_drawdown']
                })
    
    results_df = pd.DataFrame(results_data)
    
    # Print results table
    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECK RESULTS: 2016-2020")
    print("=" * 80)
    print("\nLong-Short vs Long-Only Strategy Comparison")
    print("-" * 80)
    
    for ticker in ['SPY', 'USO']:
        print(f"\n{ticker}:")
        ticker_df = results_df[results_df['Ticker'] == ticker]
        
        for _, row in ticker_df.iterrows():
            print(f"\n  {row['Model']}:")
            print(f"    Long-Short: Return {row['LS_Return']:>8.2%}, Sharpe {row['LS_Sharpe']:>6.2f}, MaxDD {row['LS_MaxDD']:>8.2%}")
            print(f"    Long-Only:  Return {row['LO_Return']:>8.2%}, Sharpe {row['LO_Sharpe']:>6.2f}, MaxDD {row['LO_MaxDD']:>8.2%}")
    
    # Save results
    output_dir = Path('data/processed/robustness')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path_ls = output_dir / 'backtest_2016_2020_longshort.pkl'
    output_path_lo = output_dir / 'backtest_2016_2020_longonly.pkl'
    output_path_table = output_dir / 'results_table_2016_2020.csv'
    
    with open(output_path_ls, 'wb') as f:
        pickle.dump(backtest_longshort, f)
    
    with open(output_path_lo, 'wb') as f:
        pickle.dump(backtest_longonly, f)
    
    results_df.to_csv(output_path_table, index=False)
    
    logger.success(f"\nResults saved:")
    logger.success(f"  Long-short: {output_path_ls}")
    logger.success(f"  Long-only:  {output_path_lo}")
    logger.success(f"  Table:      {output_path_table}")
    logger.success(f"  Log file:   {log_file}")
    
    # Compare with 2020-2024 results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH 2020-2024 RESULTS")
    logger.info("=" * 80)
    
    try:
        # Load 2020-2024 long-only results
        with open('data/processed/backtest_results_longonly_corrected.pkl', 'rb') as f:
            results_2020_2024 = pickle.load(f)
        
        print("\n" + "=" * 80)
        print("COMPARISON: 2016-2020 vs 2020-2024 (Long-Only)")
        print("=" * 80)
        
        for ticker in ['SPY', 'USO']:
            print(f"\n{ticker}:")
            
            # 2016-2020 results
            results_2016 = results_df[results_df['Ticker'] == ticker]
            
            # 2020-2024 results
            results_2024 = [
                r for r in results_2020_2024['iteration_1']
                if r['ticker'] == ticker
            ]
            
            for model_name in ['lightgbm', 'ebm', 'benchmark']:
                row_2016 = results_2016[results_2016['Model'] == model_name.upper()]
                result_2024 = next(
                    (r for r in results_2024 if r['model_name'] == model_name),
                    None
                )
                
                if not row_2016.empty and result_2024:
                    print(f"\n  {model_name.upper()}:")
                    print(f"    2016-2020: Return {row_2016.iloc[0]['LO_Return']:>8.2%}, Sharpe {row_2016.iloc[0]['LO_Sharpe']:>6.2f}")
                    print(f"    2020-2024: Return {result_2024['metrics']['total_return']:>8.2%}, Sharpe {result_2024['metrics']['sharpe']:>6.2f}")
        
        logger.info("\nComparison complete - see output above")
        
    except FileNotFoundError:
        logger.warning("Could not load 2020-2024 results for comparison")
    
    logger.info("\n" + "=" * 80)
    logger.info("ROBUSTNESS CHECK COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Training period: 2008-2016")
    logger.info(f"Test period: 2016-2020")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
