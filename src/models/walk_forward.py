"""
Walk-Forward Cross-Validation Module

Implements walk-forward validation for time series financial data.
Supports both expanding and rolling window approaches.
"""

from typing import Dict, Any, List, Tuple, Generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


class WalkForwardCV:
    """
    Walk-forward cross-validation for time series data.
    
    Trains exclusively on SPY (S&P 500 ETF) using pure technical indicators
    derived from price and volume data.
    
    Supports two window types:
    
    1. Rolling window (default):
       - Fixed-size training window (e.g., 3 years)
       - Slides forward by 1 year per fold
       - Validation window follows training window (e.g., 1 year)
       - No overlap between validation periods
       
       Example with 3-year train, 1-year val:
           Fold 1: train 2008-2010, val 2011
           Fold 2: train 2009-2011, val 2012
           Fold 3: train 2010-2012, val 2013
           ...
    
    2. Expanding window (legacy):
       - Training set grows with each fold
       - Validation window has fixed size
       
       Example with 8 splits and 1-year validation:
           Fold 1: train 2008-2011, val 2012
           Fold 2: train 2008-2012, val 2013
           ...
           Fold 8: train 2008-2019, val 2020
    
    Ensures no look-ahead bias in both modes.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WalkForwardCV.
        
        Args:
            config: Configuration dictionary with walk_forward parameters
        """
        self.config = config
        models_config = config.get('models', {})
        wf_config = models_config.get('walk_forward', {})
        
        # Date ranges
        self.train_start = pd.to_datetime(wf_config.get('train_start', '2008-01-01'))
        self.train_end = pd.to_datetime(wf_config.get('train_end', '2020-01-01'))
        self.test_start = pd.to_datetime(wf_config.get('test_start', '2020-01-01'))
        self.test_end = pd.to_datetime(wf_config.get('test_end', '2024-12-31'))
        
        # Walk-forward parameters
        self.window_type = wf_config.get('window_type', 'rolling')
        self.train_window_years = wf_config.get('train_window_years', 3)
        self.val_window_years = wf_config.get('val_window_years', 1)
        
        # Feature exclusions
        self.exclude_from_features = models_config.get('exclude_from_features', [
            'Close', 'High', 'Low', 'Open', 'Volume',
            'label', 'label_binary'
        ])
        
        logger.info("WalkForwardCV initialized")
        logger.info(f"Window type: {self.window_type}")
        logger.info(f"Training period: {self.train_start.date()} to {self.train_end.date()}")
        logger.info(f"Test period: {self.test_start.date()} to {self.test_end.date()}")
        if self.window_type == 'rolling':
            logger.info(f"Training window: {self.train_window_years} year(s)")
        logger.info(f"Validation window: {self.val_window_years} year(s)")
    
    def get_feature_names(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names.
        
        Excludes OHLCV, labels, and all external market variables (VIX, oil).
        Keeps only the 16 pure technical features derived from price and volume.
        
        Args:
            data: DataFrame with all columns
            
        Returns:
            List of feature column names (16 technical features)
        """
        # Define the 16 pure technical features to KEEP
        # Excludes VIX and oil - these are used for GMM regime detection only
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
        
        all_columns = data.columns.tolist()
        
        # Keep only technical features that exist in the data
        feature_columns = [
            col for col in all_columns 
            if col in TECHNICAL_FEATURES
        ]
        
        logger.debug(f"Total columns: {len(all_columns)}")
        logger.debug(f"Technical feature columns: {len(feature_columns)}")
        logger.debug(f"Features: {feature_columns}")
        
        return feature_columns
    
    def _filter_by_date(
        self, 
        data: pd.DataFrame, 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            data: DataFrame with MultiIndex (ticker, date)
            start_date: Start date (inclusive)
            end_date: End date (exclusive)
            
        Returns:
            Filtered DataFrame
        """
        # Get date level from MultiIndex
        dates = data.index.get_level_values('date')
        
        # Filter by date range
        mask = (dates >= start_date) & (dates < end_date)
        filtered = data[mask].copy()
        
        return filtered
    
    def _prepare_xy(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X (features) and y (target) from data.
        
        Removes rows with NaN in target or features.
        
        Args:
            data: DataFrame with features and target
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X, y) with NaN rows removed
        """
        # Extract features and target
        X = data[feature_columns].copy()
        y = data['label_binary'].copy()
        
        # Remove rows where target is NaN (label_binary NaN = label 0)
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Remove rows where any feature is NaN
        valid_mask = X.notna().all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def _get_fold_dates(self) -> List[Dict[str, pd.Timestamp]]:
        """
        Calculate date ranges for each fold.
        
        For rolling window: fixed-size training window slides forward by 1 year per fold
        For expanding window: training window grows with each fold
        
        Returns:
            List of dictionaries with train/val date ranges
        """
        folds = []
        
        if self.window_type == 'rolling':
            # Rolling window: fixed training window size, slides forward by 1 year
            fold_num = 1
            current_train_start = self.train_start
            
            while True:
                train_end = current_train_start + pd.DateOffset(years=self.train_window_years)
                val_start = train_end
                val_end = val_start + pd.DateOffset(years=self.val_window_years)
                
                # Stop if validation window exceeds train_end
                if val_end > self.train_end:
                    break
                
                folds.append({
                    'fold': fold_num,
                    'train_start': current_train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end
                })
                
                # Slide window forward by 1 year
                current_train_start += pd.DateOffset(years=1)
                fold_num += 1
            
        else:
            # Expanding window: training grows with each fold (legacy behavior)
            total_days = (self.train_end - self.train_start).days
            val_days = int(self.val_window_years * 365)
            
            available_days = total_days - val_days
            step_days = available_days // (self.n_splits - 1)
            
            for i in range(self.n_splits):
                val_start = self.train_start + pd.Timedelta(days=step_days * i + val_days)
                val_end = val_start + pd.Timedelta(days=val_days)
                
                # Last fold ends in train_end
                if i == self.n_splits - 1:
                    val_end = self.train_end
                    val_start = val_end - pd.Timedelta(days=val_days)
                
                folds.append({
                    'fold': i + 1,
                    'train_start': self.train_start,
                    'train_end': val_start,
                    'val_start': val_start,
                    'val_end': val_end
                })
        
        return folds
    
    def split(
        self, 
        data: pd.DataFrame
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate train/validation splits for walk-forward CV.
        
        Args:
            data: DataFrame with MultiIndex (ticker, date) and all columns
            
        Yields:
            Dictionary with fold information and data splits
        """
        logger.info("=" * 80)
        logger.info("WALK-FORWARD CROSS-VALIDATION SPLITS")
        logger.info("=" * 80)
        
        # Get feature columns
        feature_columns = self.get_feature_names(data)
        logger.info(f"Using {len(feature_columns)} features")
        
        # Get fold date ranges
        fold_dates = self._get_fold_dates()
        total_folds = len(fold_dates)
        
        # Generate each fold
        for fold_info in fold_dates:
            fold_num = fold_info['fold']
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"FOLD {fold_num}/{total_folds}")
            logger.info("=" * 80)
            
            # Filter data by date ranges
            train_data = self._filter_by_date(
                data, 
                fold_info['train_start'], 
                fold_info['train_end']
            )
            
            val_data = self._filter_by_date(
                data,
                fold_info['val_start'],
                fold_info['val_end']
            )
            
            logger.info(f"Train period: {fold_info['train_start'].date()} to {fold_info['train_end'].date()}")
            logger.info(f"Val period: {fold_info['val_start'].date()} to {fold_info['val_end'].date()}")
            logger.info(f"Train rows (before cleaning): {len(train_data)}")
            logger.info(f"Val rows (before cleaning): {len(val_data)}")
            
            # Prepare X, y
            X_train, y_train = self._prepare_xy(train_data, feature_columns)
            X_val, y_val = self._prepare_xy(val_data, feature_columns)
            
            logger.info(f"Train rows (after cleaning): {len(X_train)}")
            logger.info(f"Val rows (after cleaning): {len(X_val)}")
            
            # Log label distribution
            if len(y_train) > 0:
                train_dist = y_train.value_counts().sort_index()
                logger.info("Train label distribution:")
                for label, count in train_dist.items():
                    pct = (count / len(y_train)) * 100
                    logger.info(f"  Label {int(label):2d}: {count:5d} ({pct:5.2f}%)")
            
            if len(y_val) > 0:
                val_dist = y_val.value_counts().sort_index()
                logger.info("Val label distribution:")
                for label, count in val_dist.items():
                    pct = (count / len(y_val)) * 100
                    logger.info(f"  Label {int(label):2d}: {count:5d} ({pct:5.2f}%)")
            
            # Yield fold data
            # Also include full DataFrames (train_data_full, val_data_full) for regime detection
            yield {
                'fold_number': fold_num,
                'train_start': fold_info['train_start'],
                'train_end': fold_info['train_end'],
                'val_start': fold_info['val_start'],
                'val_end': fold_info['val_end'],
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'train_dates': X_train.index.get_level_values('date'),
                'val_dates': X_val.index.get_level_values('date'),
                'feature_names': feature_columns,
                'train_data_full': train_data,  # Full DataFrame with all columns including vix
                'val_data_full': val_data  # Full DataFrame with all columns including vix
            }
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("WALK-FORWARD CV COMPLETE")
        logger.info("=" * 80)
    
    def get_oos_data(
        self, 
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Get out-of-sample test data for final evaluation.
        
        Args:
            data: DataFrame with MultiIndex (ticker, date) and all columns
            
        Returns:
            Tuple of (X_test, y_test, feature_names)
        """
        logger.info("=" * 80)
        logger.info("OUT-OF-SAMPLE TEST DATA")
        logger.info("=" * 80)
        logger.info(f"Test period: {self.test_start.date()} to {self.test_end.date()}")
        
        # Get feature columns
        feature_columns = self.get_feature_names(data)
        
        # Filter test data
        test_data = self._filter_by_date(data, self.test_start, self.test_end)
        logger.info(f"Test rows (before cleaning): {len(test_data)}")
        
        # Prepare X, y
        X_test, y_test = self._prepare_xy(test_data, feature_columns)
        logger.info(f"Test rows (after cleaning): {len(X_test)}")
        
        # Log label distribution
        if len(y_test) > 0:
            test_dist = y_test.value_counts().sort_index()
            logger.info("Test label distribution:")
            for label, count in test_dist.items():
                pct = (count / len(y_test)) * 100
                logger.info(f"  Label {int(label):2d}: {count:5d} ({pct:5.2f}%)")
        
        logger.info("=" * 80)
        
        return X_test, y_test, feature_columns
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of walk-forward configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        fold_dates = self._get_fold_dates()
        
        summary = {
            'window_type': self.window_type,
            'train_start': self.train_start.strftime('%Y-%m-%d'),
            'train_end': self.train_end.strftime('%Y-%m-%d'),
            'test_start': self.test_start.strftime('%Y-%m-%d'),
            'test_end': self.test_end.strftime('%Y-%m-%d'),
            'n_splits': len(fold_dates),
            'train_window_years': self.train_window_years if self.window_type == 'rolling' else None,
            'val_window_years': self.val_window_years,
            'folds': [
                {
                    'fold': f['fold'],
                    'train_start': f['train_start'].strftime('%Y-%m-%d'),
                    'train_end': f['train_end'].strftime('%Y-%m-%d'),
                    'val_start': f['val_start'].strftime('%Y-%m-%d'),
                    'val_end': f['val_end'].strftime('%Y-%m-%d')
                }
                for f in fold_dates
            ]
        }
        
        return summary
