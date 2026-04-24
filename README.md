# Market Regime Detection and Knowledge Distillation for Financial Time Series

This repository contains the implementation of a Master's thesis on supervised machine learning for binary classification of financial time series (SPY and USO ETFs) using the Triple Barrier Method for labeling, with two iterations: a baseline model and an enhanced version incorporating GMM-based market regime detection features.

## Repository Structure

```
.
├── config/
│   └── config.yaml                          # Configuration file with all hyperparameters
├── src/
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py           # Technical indicators and feature generation
│   │   └── triple_barrier.py                # Triple Barrier Method labeling implementation
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── aligner.py                       # Data alignment across multiple sources
│   │   ├── downloader.py                    # Yahoo Finance data downloader
│   │   ├── fred_loader.py                   # FRED macroeconomic data loader
│   │   └── loader.py                        # Main data loading orchestrator
│   └── models/
│       ├── __init__.py
│       ├── backtest.py                      # Backtesting engine for OOS evaluation
│       ├── regime_detection.py              # GMM-based market regime detection
│       ├── train.py                         # Model training (LightGBM, EBM)
│       └── walk_forward.py                  # Walk-forward validation implementation
├── main.py                                  # Main pipeline: data ingestion + feature generation
├── run_walk_forward.py                      # Iteration 1: Baseline walk-forward validation
├── run_walk_forward_regime.py               # Iteration 2: Walk-forward with regime features
├── run_backtest.py                          # Main OOS backtest (2020-2024)
├── run_backtest_longonly.py                 # Long-only strategy OOS backtest
├── run_rolling_oos_evaluation.py            # Rolling OOS evaluation (annual retraining)
├── run_robustness_longonly_2016_2020.py     # Robustness check on earlier period
├── run_shap_analysis.py                     # SHAP feature importance analysis
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## How to Reproduce Results

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

**Python Version:** 3.9+

**Key Dependencies:**
- `yfinance` - Yahoo Finance data download
- `pandas-datareader` - FRED data access
- `lightgbm` - LightGBM model
- `interpret` - Explainable Boosting Machine (EBM)
- `scikit-learn` - GMM regime detection, metrics
- `shap` - SHAP feature importance
- `loguru` - Logging

### 2. Generate Features and Labels

Run the main pipeline to download data, align sources, generate technical indicators, and apply Triple Barrier labeling:

```bash
python main.py
```

**Output:**
- `data/processed/assets.h5` - HDF5 file with features and labels
- `data/processed/aligned_data_*.csv` - Aligned price data
- `logs/pipeline_*.log` - Execution log

### 3. Run Iteration 1 (Baseline)

Train LightGBM and EBM models using expanding walk-forward validation (8 folds, 2008-2020) with 23 technical features:

```bash
python run_walk_forward.py
```

**Output:**
- `data/processed/walk_forward_results.pkl` - Fold-level metrics and trained models
- `logs/walk_forward_*.log` - Training log

### 4. Run Iteration 2 (With Regime Features)

Train models with 4 additional GMM regime features (27 total features):

```bash
python run_walk_forward_regime.py
```

**Output:**
- `data/processed/walk_forward_results_regime.pkl` - Results with regime features
- `logs/walk_forward_regime_*.log` - Training log

### 5. Run Main OOS Evaluation (2020-2024)

Evaluate trained models on out-of-sample period using long-short strategy:

```bash
python run_backtest.py
```

**Output:**
- `data/processed/backtest_results.pkl` - OOS metrics (accuracy, ROC-AUC, returns, Sharpe)
- `logs/backtest_*.log` - Backtest log

### 6. Run Long-Only Strategy Evaluation

Evaluate models using long-only strategy (buy on signal=1, hold cash on signal=0):

```bash
python run_backtest_longonly.py
```

**Output:**
- `data/processed/backtest_results_longonly.pkl` - Long-only strategy metrics
- `logs/backtest_longonly_*.log` - Backtest log

### 7. Run Rolling OOS Evaluation

Evaluate models with annual retraining (2021, 2022, 2023, 2024):

```bash
python run_rolling_oos_evaluation.py
```

**Output:**
- `data/processed/rolling_oos/rolling_oos_results.pkl` - Rolling OOS metrics
- `logs/rolling_oos_*.log` - Execution log

### 8. Run Robustness Check (2016-2020)

Test models on an earlier OOS period to verify robustness:

```bash
python run_robustness_longonly_2016_2020.py
```

**Output:**
- `data/processed/robustness_2016_2020_results.pkl` - Robustness check metrics
- `logs/robustness_2016_2020_*.log` - Execution log

### 9. Run SHAP Analysis

Generate SHAP feature importance analysis for model interpretability:

```bash
python run_shap_analysis.py
```

**Output:**
- `data/processed/shap_values_*.pkl` - SHAP values for each model
- `logs/shap_analysis_*.log` - Analysis log

## Data Sources

- **Price Data:** Yahoo Finance via `yfinance` library
  - SPY (S&P 500 ETF)
  - USO (United States Oil Fund ETF)
  - ^VIX (CBOE Volatility Index)
  - CL=F (Crude Oil Futures)
  - DX-Y.NYB (US Dollar Index)

- **Macroeconomic Data:** Federal Reserve Economic Data (FRED) via `pandas-datareader`
  - Interest rates, inflation, unemployment, GDP, etc.

## Key Design Decisions

- **Triple Barrier Method:** k=1.0, max_holding=8 days, volatility=ewm(span=20).std()
- **Binary Labels:** 1=take profit, -1=everything else (stop loss or time barrier)
- **GMM Regimes:** 3 regimes ordered by volatility, retrained per fold
- **Walk-Forward Validation:** 8 folds, expanding window (2008-2020)
- **OOS Period:** 2020-2024 (main evaluation), 2016-2020 (robustness check)
- **Models:** LightGBM (gradient boosting), EBM (explainable boosting)
- **Macroeconomic Features:** FRED series were excluded from the final feature set due to frequency mismatch (monthly releases forward-filled to daily frequency). Only daily market variables are used (VIX, oil).

## Results Summary

The thesis demonstrates that:
1. Models achieve ~56% accuracy on in-sample walk-forward validation (2008-2020)
2. Performance degrades significantly on OOS period (2020-2024): ~47-52% accuracy
3. GMM regime features provide minimal improvement
4. Long-only strategies show better risk-adjusted returns than long-short
5. Annual retraining provides modest improvements but doesn't solve fundamental degradation
6. SHAP analysis reveals volatility and momentum features as most important

## License

This code is provided for academic purposes as part of a Master's thesis.

## Contact

For questions about this implementation, please refer to the thesis document or contact the author through the academic institution.
