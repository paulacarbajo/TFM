"""
Feature Engineering — computes stationary technical features for the pipeline.

Called by main.py after data alignment (Step 5), before triple-barrier labeling (Step 6).
Each ticker is processed independently to prevent cross-sectional look-ahead bias.

Active model features (10):
    Momentum  : ret_5d, ret_21d
    Volatility: vol_rel                            (vol_20d / vol_60d — relative to own history)
    Oscillator: rsi_14
    MACD      : macd_line, macd_signal             (normalised by Close — % of price)
    Bollinger : bb_pct, bb_width                   (dimensionless; no absolute band levels)
    ATR       : atr_14                             (Wilder EWM, normalised by Close — % of price)
    Volume    : volume_direction                   (volume_ratio × sign(ret_1d) — directional volume)

Note: vol_20d is computed but excluded from model features — it sets the barrier width in
triple-barrier labeling, creating a mechanical correlation with label_binary. vol_rel
(vol_20d / vol_60d) captures the same regime signal without the level bias.

Additional computed columns (not model features):
    ret_1d          — intermediate for vol_20d; also available as candidate feature
    sma_200_dist    — macro trend filter; available as candidate feature
    ret_1d_forward  — next-day return for P&L backtesting
    ret_10d_forward — 10-day forward return; available as candidate feature
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from loguru import logger


ACTIVE_FEATURES = [
    'ret_5d', 'ret_21d',                                      # momentum
    'vol_rel',                                                # volatility regime (vol_20d / vol_60d)
    'rsi_14',                                                 # oscillator
    'macd_line', 'macd_signal',                               # trend (macd_hist excluded: identity of line - signal)
    'bb_pct', 'bb_width',                                     # bands
    'atr_14',                                                 # volatility
    'volume_direction',                                       # directional volume (volume_ratio × sign(ret_1d))
    # vol_20d excluded: barrier width in triple-barrier → mechanical correlation with label_binary
]


class FeatureEngineer:
    """
    Computes stationary technical features for SPY daily OHLCV data.
    All features are dimensionless (price-scale independent).
    10 active model features — see module-level ACTIVE_FEATURES.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        features_config = config.get('features', {})
        technical_config = features_config.get('technical', {})

        self.momentum_periods = technical_config.get('momentum_periods', [1, 5, 21])
        self.volatility_period = technical_config.get('volatility_period', 20)
        self.rsi_period = technical_config.get('rsi_period', 14)
        self.macd_fast = technical_config.get('macd_fast', 12)
        self.macd_slow = technical_config.get('macd_slow', 26)
        self.macd_signal = technical_config.get('macd_signal', 9)
        self.bollinger_period = technical_config.get('bollinger_period', 20)
        self.bollinger_std = technical_config.get('bollinger_std', 2)
        self.atr_period = technical_config.get('atr_period', 14)
        self.volume_ma_period = technical_config.get('volume_ma_period', 20)

        logger.info(f"FeatureEngineer initialised  |  {len(ACTIVE_FEATURES)} active features")
        logger.info(f"Momentum periods: {self.momentum_periods}")
        logger.info(
            f"RSI: period={self.rsi_period}  |  "
            f"MACD: {self.macd_fast}/{self.macd_slow}/{self.macd_signal} (normalised by Close)  |  "
            f"ATR: period={self.atr_period} (Wilder, normalised by Close)"
        )
        logger.info(
            f"Bollinger: period={self.bollinger_period}, std={self.bollinger_std}  |  "
            f"Volume MA: {self.volume_ma_period}"
        )

    # ------------------------------------------------------------------
    # Individual indicator calculators (called per ticker)
    # ------------------------------------------------------------------

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum returns and forward returns.

        Features:    ret_1d, ret_5d, ret_21d (from momentum_periods config).
        Backtesting: ret_1d_forward (next-day P&L — NOT a model feature).
        Auxiliary:   ret_10d_forward (10-day forward return — available as candidate feature).
        Note: ret_1d is an intermediate required by calculate_volatility and calculate_volume_ratio.
        """
        for period in self.momentum_periods:
            data[f'ret_{period}d'] = data['Close'].pct_change(period)

        data['ret_1d_forward'] = data['ret_1d'].shift(-1)
        data['ret_10d_forward'] = data['Close'].pct_change(10).shift(-10)

        logger.debug(f"Calculated returns: {[f'ret_{p}d' for p in self.momentum_periods]}")
        return data

    def calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute vol_20d (EWM std of ret_1d) and vol_rel (vol_20d / vol_60d).

        vol_20d: barrier width for triple-barrier labeling (k × vol_20d). NOT a model feature.
        vol_60d: longer-horizon EWM std, same method as vol_20d.
        vol_rel: vol_20d / vol_60d — volatility relative to own 60d history.
            > 1: vol is elevated (expanding), < 1: vol is compressed (contracting).
            Captures regime signal without the level bias that creates mechanical
            correlation between raw vol_20d and label_binary.
        Requires ret_1d — must be called after calculate_returns.
        """
        if 'ret_1d' not in data.columns:
            raise ValueError(
                "'ret_1d' not found. "
                "calculate_returns must be called before calculate_volatility."
            )

        short_span = self.volatility_period                  # default 20
        long_span  = self.volatility_period * 3              # default 60

        data['vol_20d'] = data['ret_1d'].ewm(span=short_span).std()
        data['vol_60d'] = data['ret_1d'].ewm(span=long_span).std()
        data['vol_rel'] = data['vol_20d'] / data['vol_60d']

        logger.debug(
            f"Calculated vol_20d (EWM span={short_span}), "
            f"vol_60d (EWM span={long_span}), vol_rel (ratio)"
        )
        return data

    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI using Wilder's smoothing: ewm(alpha=1/n, adjust=False).
        Bounded [0, 100], dimensionless — no normalisation needed.
        """
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_period, adjust=False).mean()

        data[f'rsi_{self.rsi_period}'] = 100 - (100 / (1 + gain / loss))
        logger.debug(f"Calculated rsi_{self.rsi_period}  |  Wilder EWM(alpha=1/{self.rsi_period})")
        return data

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MACD (12/26/9) normalised by Close price (% of price).
        Raw MACD in dollars scales with price level; dividing by Close makes it
        comparable across the 2004–2024 SPY price range (~$90–$600).
        """
        ema_fast = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()

        data['macd_line'] = (ema_fast - ema_slow) / data['Close'] * 100
        data['macd_signal'] = data['macd_line'].ewm(
            span=self.macd_signal, adjust=False
        ).mean()
        data['macd_hist'] = data['macd_line'] - data['macd_signal']

        logger.debug(
            f"Calculated MACD ({self.macd_fast}/{self.macd_slow}/{self.macd_signal})  |  "
            "normalised by Close (%)"
        )
        return data

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute bb_pct and bb_width (dimensionless). Absolute band levels are discarded.

        bb_pct   = (Close - lower) / (upper - lower)  — position in bands [~0,1]
        bb_width = (upper - lower) / mid              — relative band width
        """
        bb_mid = data['Close'].rolling(window=self.bollinger_period).mean()
        bb_std = data['Close'].rolling(window=self.bollinger_period).std()
        bb_upper = bb_mid + self.bollinger_std * bb_std
        bb_lower = bb_mid - self.bollinger_std * bb_std

        data['bb_pct'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        data['bb_width'] = (bb_upper - bb_lower) / bb_mid

        logger.debug(
            f"Calculated Bollinger Bands  |  period={self.bollinger_period}, "
            f"std={self.bollinger_std}  |  bb_pct + bb_width only"
        )
        return data

    def calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ATR via Wilder's EWM (alpha=1/n), normalised by Close (ATR%).
        TrueRange = max(H-L, |H-C_prev|, |L-C_prev|).
        Normalisation by Close makes it comparable across price regimes.
        """
        high_low   = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close  = (data['Low']  - data['Close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_raw = true_range.ewm(alpha=1 / self.atr_period, adjust=False).mean()

        data[f'atr_{self.atr_period}'] = atr_raw / data['Close'] * 100

        logger.debug(
            f"Calculated atr_{self.atr_period}  |  "
            f"Wilder EWM(alpha=1/{self.atr_period}), normalised by Close (%)"
        )
        return data

    def calculate_volume_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume_ratio and volume_direction.

        volume_ratio:     volume / 20d rolling mean — measures activity magnitude.
        volume_direction: volume_ratio × sign(ret_1d) — directional volume.
            Positive = above-average volume on an up day (buyers in control).
            Negative = above-average volume on a down day (sellers in control).
            Captures "does volume confirm the move?" — the core principle of OBV.
        Requires ret_1d — must be called after calculate_returns.
        """
        volume_ma = data['Volume'].rolling(window=self.volume_ma_period).mean()
        data['volume_ratio'] = data['Volume'] / volume_ma
        data['volume_direction'] = data['volume_ratio'] * np.sign(data['ret_1d'])

        logger.debug(
            f"Calculated volume_ratio and volume_direction  |  MA period={self.volume_ma_period}"
        )
        return data

    def calculate_sma_200_dist(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute distance from 200-day SMA, normalised: (Close - SMA200) / SMA200.
        Positive = above trend (bullish context), negative = below trend (bearish).
        Macro trend filter — captures bull/bear market context beyond short-term momentum.
        """
        sma_200 = data['Close'].rolling(window=200).mean()
        data['sma_200_dist'] = (data['Close'] - sma_200) / sma_200

        logger.debug("Calculated sma_200_dist")
        return data

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def engineer_features_for_ticker(self, ticker_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature calculations to a single ticker.
        calculate_returns must precede calculate_volatility (vol_20d depends on ret_1d).
        """
        result = ticker_data.copy()

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
        Apply feature engineering to all tickers in the aligned dataset.
        Each ticker is processed independently to prevent cross-sectional look-ahead bias.
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 60)
        logger.info(f"Input shape: {data.shape}")

        tickers = data.index.get_level_values('ticker').unique()
        logger.info(f"Processing {len(tickers)} ticker(s): {tickers.tolist()}")

        ticker_results = []

        for ticker in tickers:
            logger.info(f"  Processing {ticker}")
            ticker_data = data.xs(ticker, level='ticker')
            ticker_features = self.engineer_features_for_ticker(ticker_data)
            ticker_features['ticker'] = ticker
            ticker_features = ticker_features.reset_index().set_index(['ticker', 'date'])
            ticker_results.append(ticker_features)
            logger.success(f"  {ticker}: {ticker_features.shape}")

        result = pd.concat(ticker_results).sort_index()

        new_cols = sorted(set(result.columns) - set(data.columns))
        logger.info("=" * 60)
        logger.success("FEATURE ENGINEERING COMPLETE")
        logger.info(f"Active features ({len(ACTIVE_FEATURES)}): {ACTIVE_FEATURES}")
        logger.info(f"All new columns ({len(new_cols)}): {new_cols}")
        logger.info(f"Output shape: {result.shape}")
        logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Inspection utility
    # ------------------------------------------------------------------

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return quality summary classifying columns into OHLCV, FRED, auxiliary, and features."""
        ohlcv_cols = {'Close', 'High', 'Low', 'Open', 'Volume'}
        auxiliary_cols = {
            'ret_1d_forward', 'ret_10d_forward',
        }
        label_cols = {'label', 'label_binary'}

        fred_base = list(self.config.get('fred_series', {}).keys())
        fred_cols = set()
        for base in fred_base:
            fred_cols.update({base, f'{base}_diff', f'{base}_chg'})

        exclude_cols = ohlcv_cols | fred_cols | auxiliary_cols | label_cols
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        return {
            'total_columns': len(data.columns),
            'active_feature_count': len(ACTIVE_FEATURES),
            'detected_feature_columns': len(feature_cols),
            'features': feature_cols,
            'missing_values_in_features': int(
                data[feature_cols].isnull().sum().sum()
            ),
            'shape': data.shape,
        }
