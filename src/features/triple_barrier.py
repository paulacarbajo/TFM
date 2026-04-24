"""
Triple Barrier Labeling Module

Implements triple barrier method for labeling financial time series data.
Based on López de Prado's methodology for meta-labeling.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class TripleBarrierLabeler:
    """
    Labels financial data using the triple barrier method.
    
    For each observation at time t, defines three barriers:
    - Upper barrier: price * (1 + k * volatility)
    - Lower barrier: price * (1 - k * volatility)
    - Time barrier: max_holding_period days
    
    Ternary label is determined by which barrier is touched first:
    - Upper touched first → label = 1 (take profit)
    - Lower touched first → label = -1 (stop loss)
    - Time expires → label = 0 (no clear signal)
    
    Binary label collapses the ternary label:
    - label = 1 → label_binary = 1 (take profit reached)
    - label = -1 or 0 → label_binary = -1 (stop loss or time barrier)
    
    All calculations are done per ticker to avoid look-ahead bias.
    Volatility is computed as ewm(span=20).std() of daily returns.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        max_holding_period (int): Maximum holding period in days (time barrier)
        vol_multiplier (float): Volatility multiplier k for barrier width
        min_ret (float): Minimum return threshold
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TripleBarrierLabeler.
        
        Args:
            config: Configuration dictionary with labeling parameters
        """
        self.config = config
        features_config = config.get('features', {})
        tb_config = features_config.get('triple_barrier', {})
        
        self.max_holding_period = tb_config.get('max_holding_period', 8)
        self.vol_multiplier = tb_config.get('vol_multiplier', 1.0)
        self.min_ret = tb_config.get('min_ret', 0.0)
        
        logger.info("TripleBarrierLabeler initialized")
        logger.info(f"Max holding period: {self.max_holding_period} days")
        logger.info(f"Volatility multiplier (k): {self.vol_multiplier}")
        logger.info(f"Minimum return threshold: {self.min_ret}")
    
    def get_barrier_for_observation(
        self,
        close_prices: pd.Series,
        current_idx: int,
        current_price: float,
        volatility: float
    ) -> int:
        """
        Determine which barrier is touched first for a single observation.
        
        Non-vectorized reference implementation for validation and debugging.
        The main labeling pipeline uses label_ticker_data for performance.
        Returns the ternary label (1, -1, 0) — same logic as label_ticker_data
        before binary collapse.
        
        Args:
            close_prices: Series of close prices
            current_idx: Index of current observation
            current_price: Current close price
            volatility: Current volatility (ewm span=20 std of returns)
            
        Returns:
            Label: 1 (upper barrier), -1 (lower barrier), 0 (time expired)
        """
        threshold = max(volatility * self.vol_multiplier, self.min_ret)
        
        upper_barrier = current_price * (1 + threshold)
        lower_barrier = current_price * (1 - threshold)
        
        end_idx = min(current_idx + self.max_holding_period, len(close_prices))
        future_prices = close_prices.iloc[current_idx + 1:end_idx]
        
        if len(future_prices) == 0:
            return 0
        
        for price in future_prices:
            if price >= upper_barrier:
                return 1
            elif price <= lower_barrier:
                return -1
        
        return 0
    
    def label_ticker_data(self, ticker_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple barrier labeling to a single ticker's data.
        
        Vectorized implementation using numpy for performance.
        
        Produces two label columns:
        - label: ternary (1 = take profit, -1 = stop loss, 0 = time barrier)
        - label_binary: binary (1 = take profit, -1 = stop loss or time barrier)
        
        Args:
            ticker_data: DataFrame with Close prices and vol_20d column
            
        Returns:
            DataFrame with 'label', 'label_binary', and 'days_to_barrier' columns added
        """
        result = ticker_data.copy()
        
        vol_col = 'vol_20d'
        if vol_col not in result.columns:
            raise ValueError(
                f"Volatility column '{vol_col}' not found. "
                f"Ensure feature engineering has been run before labeling. "
                f"Available columns: {result.columns.tolist()}"
    )
        
        close = result['Close'].values
        vols = result[vol_col].values
        n = len(close)
        
        labels = np.zeros(n)
        days_to_barrier = np.zeros(n)
        
        take_profit_count = 0
        stop_loss_count = 0
        time_barrier_count = 0
        
        for i in range(n):
            if np.isnan(vols[i]) or np.isnan(close[i]):
                labels[i] = np.nan
                days_to_barrier[i] = np.nan
                continue
            
            threshold = max(vols[i] * self.vol_multiplier, self.min_ret)
            upper = close[i] * (1 + threshold)
            lower = close[i] * (1 - threshold)
            
            end = min(i + self.max_holding_period + 1, n)
            future = close[i+1:end]
            
            if len(future) == 0:
                labels[i] = 0
                days_to_barrier[i] = 0
                time_barrier_count += 1
                continue
            
            up_cross = np.where(future >= upper)[0]
            dn_cross = np.where(future <= lower)[0]
            
            first_up = up_cross[0] if len(up_cross) > 0 else n
            first_dn = dn_cross[0] if len(dn_cross) > 0 else n
            
            if first_up == n and first_dn == n:
                # Time barrier: price never reached either barrier
                labels[i] = 0
                days_to_barrier[i] = len(future)
                time_barrier_count += 1
            elif first_up <= first_dn:
                # Take profit hit first
                labels[i] = 1
                days_to_barrier[i] = first_up + 1
                take_profit_count += 1
            else:
                # Stop loss hit first
                labels[i] = -1
                days_to_barrier[i] = first_dn + 1
                stop_loss_count += 1
        
        # Ternary label
        result['label'] = labels
        
        # Binary label: collapse 0 (time barrier) into -1
        label_binary = np.where(labels == 1, 1, -1)
        label_binary = np.where(np.isnan(labels), np.nan, label_binary)
        result['label_binary'] = label_binary
        
        result['days_to_barrier'] = days_to_barrier
        
        valid_labels = ~np.isnan(labels)
        total_valid = int(valid_labels.sum())
        
        if total_valid > 0:
            result._label_stats = {
                'take_profit': take_profit_count,
                'stop_loss': stop_loss_count,
                'time_barrier': time_barrier_count,
                'total_valid': total_valid,
                'avg_days_to_barrier': float(np.nanmean(days_to_barrier))
            }
        
        return result
    
    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple barrier labeling to all tickers in the dataset.
        
        Processes each ticker independently to avoid look-ahead bias.
        
        Args:
            data: DataFrame with MultiIndex (ticker, date) and features
            
        Returns:
            DataFrame with 'label', 'label_binary', and 'days_to_barrier' columns added
        """
        logger.info("=" * 60)
        logger.info("TRIPLE BARRIER LABELING")
        logger.info("=" * 60)
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Max holding period: {self.max_holding_period} days")
        logger.info(f"Volatility multiplier (k): {self.vol_multiplier}")
        
        tickers = data.index.get_level_values('ticker').unique()
        logger.info(f"Processing {len(tickers)} tickers: {tickers.tolist()}")
        
        ticker_results = []
        
        total_take_profit = 0
        total_stop_loss = 0
        total_time_barrier = 0
        total_valid = 0
        all_days_to_barrier = []
        
        for ticker in tickers:
            logger.info(f"\nLabeling ticker: {ticker}")
            
            ticker_data = data.xs(ticker, level='ticker')
            ticker_labeled = self.label_ticker_data(ticker_data)
            
            if hasattr(ticker_labeled, '_label_stats'):
                stats = ticker_labeled._label_stats
                tp = stats['take_profit']
                sl = stats['stop_loss']
                tb = stats['time_barrier']
                valid = stats['total_valid']
                avg_days = stats['avg_days_to_barrier']
                
                total_take_profit += tp
                total_stop_loss += sl
                total_time_barrier += tb
                total_valid += valid
                
                logger.info(f"{ticker} label statistics:")
                logger.info(f"  Take profit  (label= 1):  {tp:5d} ({tp/valid*100:5.2f}%)")
                logger.info(f"  Stop loss    (label=-1):  {sl:5d} ({sl/valid*100:5.2f}%)")
                logger.info(f"  Time barrier (label= 0):  {tb:5d} ({tb/valid*100:5.2f}%)")
                logger.info(f"  Total valid:              {valid:5d}")
                logger.info(f"  Avg days to barrier:      {avg_days:.2f}")
                
                days_col = ticker_labeled['days_to_barrier'].dropna()
                all_days_to_barrier.extend(days_col.tolist())
                
                time_barrier_pct = (tb / valid * 100) if valid > 0 else 0
                if time_barrier_pct > 40:
                    logger.warning(
                        f"{ticker}: Time barrier hit in {time_barrier_pct:.1f}% of cases. "
                        f"Consider increasing k or max_holding_period."
                    )
            
            ticker_labeled['ticker'] = ticker
            ticker_labeled = ticker_labeled.reset_index().set_index(['ticker', 'date'])
            ticker_results.append(ticker_labeled)
            
            logger.success(f"Completed {ticker}")
        
        result = pd.concat(ticker_results)
        result = result.sort_index()
        
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL STATISTICS")
        logger.info("=" * 60)
        
        if total_valid > 0:
            tp_pct = (total_take_profit / total_valid) * 100
            sl_pct = (total_stop_loss / total_valid) * 100
            tb_pct = (total_time_barrier / total_valid) * 100
            avg_days_overall = np.mean(all_days_to_barrier) if all_days_to_barrier else 0
            
            logger.info(f"Total valid observations: {total_valid}")
            logger.info(f"Take profit  (label= 1): {total_take_profit:6d} ({tp_pct:5.2f}%)")
            logger.info(f"Stop loss    (label=-1): {total_stop_loss:6d} ({sl_pct:5.2f}%)")
            logger.info(f"Time barrier (label= 0): {total_time_barrier:6d} ({tb_pct:5.2f}%)")
            logger.info(f"Average days to barrier: {avg_days_overall:.2f}")
            
            logger.info("\nBinary label distribution (models use this):")
            binary_counts = result['label_binary'].value_counts().sort_index()
            valid_binary = result['label_binary'].notna().sum()
            for val, count in binary_counts.items():
                if pd.notna(val):
                    pct = count / valid_binary * 100
                    logger.info(f"  label_binary {int(val):2d}: {count:6d} ({pct:5.2f}%)")
            
            if tb_pct > 40:
                logger.warning(
                    f"Time barrier hit in {tb_pct:.1f}% of cases. "
                    f"Current parameters: k={self.vol_multiplier}, "
                    f"max_holding={self.max_holding_period}."
                )
            else:
                logger.success(
                    f"Time barrier at {tb_pct:.1f}% — within acceptable range (<40%)"
                )
        
        logger.info("=" * 60)
        logger.success("TRIPLE BARRIER LABELING COMPLETE")
        logger.info(f"Output shape: {result.shape}")
        logger.info("=" * 60)
        
        return result
    
    def get_label_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of label distribution.
        
        Args:
            data: DataFrame with 'label' and 'label_binary' columns
            
        Returns:
            Dictionary with ternary and binary label distributions per ticker
        """
        summary = {
            'total_observations': len(data),
            'ternary_label_distribution': data['label'].value_counts().to_dict(),
            'binary_label_distribution': data['label_binary'].value_counts().to_dict(),
            'ternary_label_percentages': (
                data['label'].value_counts() / len(data) * 100
            ).to_dict(),
            'missing_labels': int(data['label'].isna().sum()),
        }
        
        tickers = data.index.get_level_values('ticker').unique()
        summary['per_ticker'] = {}
        
        for ticker in tickers:
            ticker_data = data.xs(ticker, level='ticker')
            summary['per_ticker'][ticker] = {
                'total': len(ticker_data),
                'ternary_distribution': ticker_data['label'].value_counts().to_dict(),
                'binary_distribution': ticker_data['label_binary'].value_counts().to_dict()
            }
        
        return summary