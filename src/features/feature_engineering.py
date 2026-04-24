"""
Feature Engineering Module

Calculates technical indicators and features for financial data.
All calculations are done per ticker to avoid mixing data across assets.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:
    """
    Calculates technical features for financial time series data.

    All features are calculated per ticker independently to maintain
    data integrity and avoid look-ahead bias across different assets.

    Technical indicators use standard parameterizations from the literature:
    - Momentum: 1, 5, 21 day returns
    - Volatility: EWM std with span=20 (as recommended for triple barrier)
    - RSI: 14-period (Wilder)
    - MACD: 12/26/9 (Appel)
    - Bollinger Bands: 20-period SMA ± 2 std (standard definition)
    - ATR: 14-period (Wilder)
    - Volume ratio: current volume / 20-day average

    Note: Volatility uses EWM (exponentially weighted) per tutor recommendation
    for consistency with the triple barrier labeling. Bollinger Bands use
    standard rolling window as per their original definition.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FeatureEngineer.

        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        features_config = config.get('features', {})
        technical_config = features_config.get('technical', {})

        # Momentum parameters
        self.momentum_periods = technical_config.get('momentum_periods', [1, 5, 21])

        # Volatility parameters
        self.volatility_period = technical_config.get('volatility_period', 20)

        # RSI parameters
        self.rsi_period = technical_config.get('rsi_period', 14)

        # MACD parameters
        self.macd_fast = technical_config.get('macd_fast', 12)
        self.macd_slow = technical_config.get('macd_slow', 26)
        self.macd_signal = technical_config.get('macd_signal', 9)

        # Bollinger Bands parameters
        self.bollinger_period = technical_config.get('bollinger_period', 20)
        self.bollinger_std = technical_config.get('bollinger_std', 2)

        # ATR parameters
        self.atr_period = technical_config.get('atr_period', 14)

        # Volume parameters
        self.volume_ma_period = technical_config.get('volume_ma_period', 20)

        logger.info("FeatureEngineer initialized")
        logger.info(f"Momentum periods: {self.momentum_periods}")
        logger.info(f"Volatility period (EWM span): {self.volatility_period}")
        logger.info(f"RSI period: {self.rsi_period}")
        logger.info(f"MACD: fast={self.macd_fast}, slow={self.macd_slow}, signal={self.macd_signal}")
        logger.info(f"Bollinger Bands: period={self.bollinger_period}, std={self.bollinger_std}")
        logger.info(f"ATR period: {self.atr_period}")
        logger.info(f"Volume MA period: {self.volume_ma_period}")

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum returns for different periods.
        
        Also calculates ret_1d_forward for backtesting. This column represents
        the return that would be realized if entering a position at today's close
        and exiting at tomorrow's close. It is used exclusively for computing
        strategy returns in backtesting and must never be used as a model feature
        (to avoid look-ahead bias).

        Args:
            data: DataFrame with Close prices (single ticker)

        Returns:
            DataFrame with ret_1d, ret_5d, ret_21d, ret_1d_forward columns added
        """
        result = data.copy()

        for period in self.momentum_periods:
            col_name = f'ret_{period}d'
            result[col_name] = result['Close'].pct_change(period)
            logger.debug(f"Calculated {col_name}")

        # Calculate forward return for backtesting (next day's realized return)
        # This is shift(-1) of ret_1d, representing tomorrow's return
        result['ret_1d_forward'] = result['ret_1d'].shift(-1)
        logger.debug("Calculated ret_1d_forward (next day's return for backtesting)")

        return result

    def calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate exponentially weighted volatility of daily returns.

        Uses EWM std with span=20 per tutor recommendation, consistent
        with the volatility measure used in triple barrier labeling.

        Requires ret_1d column — must be called after calculate_returns.

        Args:
            data: DataFrame with ret_1d column (single ticker)

        Returns:
            DataFrame with vol_20d column added
        """
        result = data.copy()

        if 'ret_1d' not in result.columns:
            raise ValueError(
                "ret_1d column not found. "
                "calculate_returns must be called before calculate_volatility."
            )

        result[f'vol_{self.volatility_period}d'] = result['ret_1d'].ewm(
            span=self.volatility_period
        ).std()
        logger.debug(f"Calculated vol_{self.volatility_period}d using EWM(span={self.volatility_period})")

        return result

    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            data: DataFrame with Close prices (single ticker)

        Returns:
            DataFrame with rsi_14 column added
        """
        result = data.copy()

        delta = result['Close'].diff()

        gain = delta.clip(lower=0).ewm(
            alpha=1/self.rsi_period,
            adjust=False
        ).mean()
        loss = (-delta.clip(upper=0)).ewm(
            alpha=1/self.rsi_period,
            adjust=False
        ).mean()

        rs = gain / loss
        result[f'rsi_{self.rsi_period}'] = 100 - (100 / (1 + rs))

        logger.debug(f"Calculated rsi_{self.rsi_period}")

        return result

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD 

        Args:
            data: DataFrame with Close prices (single ticker)

        Returns:
            DataFrame with macd_line, macd_signal, macd_hist columns added
        """
        result = data.copy()

        ema_fast = result['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = result['Close'].ewm(span=self.macd_slow, adjust=False).mean()

        result['macd_line'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd_line'].ewm(
            span=self.macd_signal, adjust=False
        ).mean()
        result['macd_hist'] = result['macd_line'] - result['macd_signal']

        logger.debug("Calculated MACD (line, signal, hist)")

        return result

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using standard rolling window definition.

        Args:
            data: DataFrame with Close prices (single ticker)

        Returns:
            DataFrame with bb_mid, bb_upper, bb_lower, bb_pct, bb_width columns added
        """
        result = data.copy()

        result['bb_mid'] = result['Close'].rolling(window=self.bollinger_period).mean()
        bb_std = result['Close'].rolling(window=self.bollinger_period).std()

        result['bb_upper'] = result['bb_mid'] + (self.bollinger_std * bb_std)
        result['bb_lower'] = result['bb_mid'] - (self.bollinger_std * bb_std)

        # Position within bands: 0 = at lower band, 1 = at upper band
        result['bb_pct'] = (result['Close'] - result['bb_lower']) / (
            result['bb_upper'] - result['bb_lower']
        )

        # Relative band width: measures volatility expansion/contraction
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_mid']

        logger.debug("Calculated Bollinger Bands (mid, upper, lower, pct, width)")

        return result

    def calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR (Average True Range) 

        Args:
            data: DataFrame with High, Low, Close prices (single ticker)

        Returns:
            DataFrame with atr_14 column added
        """
        result = data.copy()

        high_low = result['High'] - result['Low']
        high_close = (result['High'] - result['Close'].shift()).abs()
        low_close = (result['Low'] - result['Close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        result[f'atr_{self.atr_period}'] = true_range.rolling(
            window=self.atr_period
        ).mean()

        logger.debug(f"Calculated atr_{self.atr_period}")

        return result

    def calculate_volume_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume ratio as current volume divided by 20-day average.

        A ratio above 1 indicates above-average trading activity.

        Args:
            data: DataFrame with Volume column (single ticker)

        Returns:
            DataFrame with volume_ratio column added
        """
        result = data.copy()

        volume_ma = result['Volume'].rolling(window=self.volume_ma_period).mean()
        result['volume_ratio'] = result['Volume'] / volume_ma

        logger.debug(f"Calculated volume_ratio (MA period={self.volume_ma_period})")

        return result

    def calculate_sma_200_dist(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate distance from 200-day Simple Moving Average.

        This measures the percentage deviation of the current price from
        its 200-day SMA, a common trend indicator in technical analysis.

        Positive values indicate price is above the long-term trend (bullish).
        Negative values indicate price is below the long-term trend (bearish).

        Args:
            data: DataFrame with Close prices (single ticker)

        Returns:
            DataFrame with sma_200_dist column added
        """
        result = data.copy()

        sma_200 = result['Close'].rolling(window=200).mean()
        result['sma_200_dist'] = (result['Close'] - sma_200) / sma_200

        logger.debug("Calculated sma_200_dist (distance from 200-day SMA)")

        return result

    def engineer_features_for_ticker(self, ticker_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for a single ticker.

        Order of calculation matters: returns must be computed before
        volatility since vol_20d depends on ret_1d.

        Args:
            ticker_data: DataFrame with OHLCV data for one ticker

        Returns:
            DataFrame with all technical features added
        """
        logger.debug(f"Engineering features for ticker data with shape {ticker_data.shape}")

        result = ticker_data.copy()

        # Order matters: returns must be calculated before volatility
        result = self.calculate_returns(result)
        result = self.calculate_volatility(result)
        result = self.calculate_rsi(result)
        result = self.calculate_macd(result)
        result = self.calculate_bollinger_bands(result)
        result = self.calculate_atr(result)
        result = self.calculate_volume_ratio(result)
        result = self.calculate_sma_200_dist(result)

        return result

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for all tickers in the dataset.

        Processes each ticker independently to avoid look-ahead bias
        across different assets.

        Args:
            data: DataFrame with MultiIndex (ticker, date) and OHLCV columns

        Returns:
            DataFrame with all technical features added
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 60)
        logger.info(f"Input data shape: {data.shape}")

        tickers = data.index.get_level_values('ticker').unique()
        logger.info(f"Processing {len(tickers)} tickers: {tickers.tolist()}")

        ticker_results = []

        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")

            ticker_data = data.xs(ticker, level='ticker')
            ticker_features = self.engineer_features_for_ticker(ticker_data)

            ticker_features['ticker'] = ticker
            ticker_features = ticker_features.reset_index().set_index(['ticker', 'date'])

            ticker_results.append(ticker_features)
            logger.success(f"Completed {ticker}: {ticker_features.shape}")

        result = pd.concat(ticker_results)
        result = result.sort_index()

        original_cols = set(data.columns)
        new_cols = set(result.columns) - original_cols

        logger.info("=" * 60)
        logger.success("FEATURE ENGINEERING COMPLETE")
        logger.info(f"Original columns: {len(original_cols)}")
        logger.info(f"New features added: {len(new_cols)}")
        logger.info(f"New features: {sorted(new_cols)}")
        logger.info(f"Output shape: {result.shape}")
        logger.info("=" * 60)

        return result

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of engineered features.

        FRED columns are read dynamically from config to avoid hardcoding.

        Args:
            data: DataFrame with features

        Returns:
            Dictionary with feature summary
        """
        ohlcv_cols = {'Close', 'High', 'Low', 'Open', 'Volume'}

        # Read FRED series names dynamically from config
        fred_base = list(self.config.get('fred_series', {}).keys())
        fred_cols = set()
        for base in fred_base:
            fred_cols.add(base)
            fred_cols.add(f'{base}_diff')
            fred_cols.add(f'{base}_chg')

        exclude_cols = ohlcv_cols | fred_cols
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        summary = {
            'total_columns': len(data.columns),
            'feature_columns': len(feature_cols),
            'features': feature_cols,
            'missing_values': int(data[feature_cols].isnull().sum().sum()),
            'shape': data.shape
        }

        return summary
