"""
Backtesting Module

Evaluates trained models on the out-of-sample (OOS) period 2020-2024.
Uses the last walk-forward fold's models, which were trained on the largest
available training set (2008-2019), to generate predictions on unseen data.

Results are computed per ticker (SPY and USO independently) and compared
against a buy-and-hold benchmark.
"""

from typing import Dict, Any, List, Optional
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss
)


class Backtester:
    """
    Backtests trained models on the out-of-sample period 2020-2024.

    Uses the last fold's models (trained on the most data) to predict
    on the OOS period. Results are computed per ticker and compared
    against a buy-and-hold benchmark.

    Classification metrics follow the convention that higher is better,
    except brier_score where lower is better (0 = perfect calibration).

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        oos_start (str): OOS period start date
        oos_end (str): OOS period end date
        exclude_cols (List[str]): Columns to exclude from features
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Backtester.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        models_config = config.get('models', {})
        wf_config = models_config.get('walk_forward', {})

        self.oos_start = wf_config.get('test_start', '2020-01-01')
        self.oos_end = wf_config.get('test_end', '2024-12-31')
        self.exclude_cols = models_config.get('exclude_from_features', [])

        logger.info("Backtester initialized")
        logger.info(f"OOS period: {self.oos_start} to {self.oos_end}")

    def _get_feature_names(self, data: pd.DataFrame) -> List[str]:
        """
        Get feature column names excluding OHLCV and label columns.

        Args:
            data: DataFrame with all columns

        Returns:
            List of feature column names
        """
        return [col for col in data.columns if col not in self.exclude_cols]

    def _filter_oos_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data for the OOS period.

        Args:
            data: Full DataFrame with MultiIndex (ticker, date)

        Returns:
            Filtered DataFrame for the OOS period
        """
        dates = data.index.get_level_values('date')
        mask = (dates >= self.oos_start) & (dates <= self.oos_end)
        oos_data = data[mask].copy()

        logger.info(f"OOS data: {len(oos_data)} rows")
        logger.info(
            f"Date range: {dates[mask].min()} to {dates[mask].max()}"
        )

        return oos_data

    def _calculate_trading_metrics(
        self,
        returns: pd.Series,
        signals: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics from signals and actual returns.

        Strategy return on each day is signal * actual_return:
        - Correct long (signal=1, return>0): profit
        - Correct short (signal=-1, return<0): profit
        - Incorrect prediction: loss

        Sharpe ratio is annualised assuming 252 trading days.
        Calmar ratio is total return divided by maximum drawdown magnitude.
        Win rate is the fraction of days with positive strategy return.

        Args:
            returns: Series of actual daily returns (ret_1d)
            signals: Series of predicted signals in {-1, 1}

        Returns:
            Dictionary of trading metric names to values
        """
        strategy_returns = signals * returns

        cumulative = (1 + strategy_returns).cumprod()
        total_return = float(cumulative.iloc[-1] - 1)

        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        calmar = abs(total_return) / abs(max_drawdown) if max_drawdown != 0 else 0.0

        positive = strategy_returns[strategy_returns > 0]
        negative = strategy_returns[strategy_returns < 0]

        return {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'calmar_ratio': float(calmar),
            'win_rate': float(len(positive) / len(strategy_returns)) if len(strategy_returns) > 0 else 0.0,
            'avg_win': float(positive.mean()) if len(positive) > 0 else 0.0,
            'avg_loss': float(negative.mean()) if len(negative) > 0 else 0.0,
            'n_long': int((signals == 1).sum()),
            'n_short': int((signals == -1).sum()),
            'cumulative_returns': cumulative
        }

    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true: pd.Series,
        returns: pd.Series,
        model_name: str,
        ticker: str,
        strategy: str = 'long_short',
        threshold: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single model on OOS data for one ticker.

        Args:
            model: Trained model
            X: Features without MultiIndex
            y_true: True labels in {-1, 1}
            returns: Actual daily returns (ret_1d)
            model_name: Model identifier
            ticker: Ticker symbol
            strategy: Trading strategy ('long_short' or 'long_only')
            threshold: Confidence threshold for predictions (default 0.5)

        Returns:
            Dictionary with evaluation results, or None if evaluation fails
        """
        try:
            y_proba = model.predict_proba(X)[:, 1]

            y_pred_binary = (y_proba > threshold).astype(int)
            
            # Generate signals based on strategy
            if strategy == 'long_only':
                y_pred_signal = y_pred_binary  # 1 or 0
            else:  # long_short
                y_pred_signal = np.where(y_pred_binary == 1, 1, -1)
            
            y_true_binary = (y_true == 1).astype(int).values

            classification = {
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'roc_auc': roc_auc_score(y_true_binary, y_proba),
                'brier_score': brier_score_loss(y_true_binary, y_proba)
            }

            trading = self._calculate_trading_metrics(
                returns,
                pd.Series(y_pred_signal, index=returns.index)
            )

            logger.success(
                f"{model_name} ({ticker}): "
                f"Acc={classification['accuracy']:.3f}, "
                f"ROC-AUC={classification['roc_auc']:.3f}, "
                f"Sharpe={trading['sharpe']:.2f}, "
                f"Return={trading['total_return']:.2%}"
            )

            return {
                'model_name': model_name,
                'ticker': ticker,
                'n_samples': len(X),
                'classification': classification,
                'trading': {k: v for k, v in trading.items() if k != 'cumulative_returns'},
                'cumulative_returns': trading['cumulative_returns'],
                'predictions': y_pred_signal,
                'probabilities': y_proba
            }

        except Exception as e:
            logger.error(f"Error evaluating {model_name} ({ticker}): {str(e)}")
            return None

    def _evaluate_benchmark(
        self,
        returns: pd.Series,
        ticker: str
    ) -> Dict[str, Any]:
        """
        Evaluate buy-and-hold benchmark (always long, signal=1).

        Args:
            returns: Actual daily returns (ret_1d)
            ticker: Ticker symbol

        Returns:
            Dictionary with benchmark trading metrics
        """
        signals = pd.Series(1, index=returns.index)
        trading = self._calculate_trading_metrics(returns, signals)

        logger.info(
            f"Benchmark ({ticker}): "
            f"Sharpe={trading['sharpe']:.2f}, "
            f"Return={trading['total_return']:.2%}"
        )

        return {
            'model_name': 'benchmark',
            'ticker': ticker,
            'n_samples': len(returns),
            'classification': {},
            'trading': {k: v for k, v in trading.items() if k != 'cumulative_returns'},
            'cumulative_returns': trading['cumulative_returns'],
            'predictions': signals.values,
            'probabilities': None
        }

    def run_backtest(
        self,
        all_fold_results: List[Dict[str, Any]],
        data: pd.DataFrame,
        strategy: str = 'long_short',
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run backtest on OOS period using the last fold's models.

        The last fold's models are used because they were trained on the
        largest available training set (all data up to 2019). Features used
        must match those in the fold's feature_names list.

        Args:
            all_fold_results: List of fold results from train_all_folds()
            data: Full DataFrame with MultiIndex (ticker, date)
            strategy: Trading strategy ('long_short' or 'long_only')
            threshold: Confidence threshold for predictions (default 0.5)

        Returns:
            Dictionary with backtest results, equity curves, and metadata
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("BACKTESTING ON OUT-OF-SAMPLE PERIOD")
        logger.info("=" * 80)
        logger.info(f"Period: {self.oos_start} to {self.oos_end}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Threshold: {threshold}")

        last_fold = all_fold_results[-1]
        fold_number = last_fold['fold_number']
        models = last_fold['models']
        feature_names = last_fold['feature_names']

        logger.info(f"Using models from fold {fold_number}")
        logger.info(f"Features: {len(feature_names)}")

        oos_data = self._filter_oos_data(data)

        # Add ticker_id before extracting features (must match walk_forward.py)
        ticker_values = oos_data.index.get_level_values('ticker')
        oos_data['ticker_id'] = (ticker_values == 'USO').astype(int)

        X_oos = oos_data[feature_names].copy()
        y_oos = oos_data['label_binary'].copy()
        returns_oos = oos_data['ret_1d'].copy()

        tickers = oos_data.index.get_level_values('ticker').unique().tolist()
        logger.info(f"Tickers: {tickers}")

        results = []

        for ticker in tickers:
            logger.info(f"\nEvaluating on {ticker}...")

            ticker_mask = oos_data.index.get_level_values('ticker') == ticker
            X_ticker = X_oos[ticker_mask].reset_index(drop=True)
            y_ticker = y_oos[ticker_mask].reset_index(drop=True)
            returns_ticker = returns_oos[ticker_mask].reset_index(drop=True)

            logger.info(f"{ticker}: {len(X_ticker)} samples")

            for model_name, model in models.items():
                if model is None:
                    logger.info(f"Skipping {model_name} (not available)")
                    continue

                result = self._evaluate_model(
                    model, X_ticker, y_ticker,
                    returns_ticker, model_name, ticker,
                    strategy=strategy,
                    threshold=threshold
                )
                if result is not None:
                    results.append(result)

            benchmark_result = self._evaluate_benchmark(returns_ticker, ticker)
            results.append(benchmark_result)

        logger.info("")
        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Evaluated {len(results)} model-ticker combinations")
        logger.info("=" * 80)

        return {
            'results': results,
            'oos_start': self.oos_start,
            'oos_end': self.oos_end,
            'fold_number': fold_number,
            'feature_names': feature_names,
            'tickers': tickers
        }

    def get_equity_curves(
        self,
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, pd.Series]]:
        """
        Extract equity curves (cumulative return series) per ticker and model.

        Args:
            backtest_results: Results from run_backtest()

        Returns:
            Nested dictionary: {ticker: {model_name: cumulative_returns_series}}
        """
        curves = {}

        for ticker in backtest_results['tickers']:
            curves[ticker] = {}
            ticker_results = [
                r for r in backtest_results['results'] if r['ticker'] == ticker
            ]
            for result in ticker_results:
                model_name = result['model_name']
                curves[ticker][model_name] = result['cumulative_returns']

        return curves

    def get_backtest_table(
        self,
        backtest_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Build a summary table of backtest results.

        Returns a DataFrame indexed by (ticker, model) with all
        classification and trading metrics as columns.
        Note: brier_score is lower-is-better; all other metrics are higher-is-better.

        Args:
            backtest_results: Results from run_backtest()

        Returns:
            DataFrame indexed by (ticker, model)
        """
        rows = []

        for result in backtest_results['results']:
            row = {
                'model': result['model_name'],
                'ticker': result['ticker'],
                'n_samples': result['n_samples']
            }

            if result['classification']:
                row.update(result['classification'])

            row.update(result['trading'])
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index(['ticker', 'model'])

        return df

    def save_results(
        self,
        backtest_results: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> None:
        """
        Save backtest results to a pickle file.

        Args:
            backtest_results: Results from run_backtest()
            output_path: Output path (defaults to data/processed/backtest_results.pkl)
        """
        if output_path is None:
            output_path = Path('data/processed/backtest_results.pkl')

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(backtest_results, f)

        logger.success(f"Backtest results saved to {output_path}")
