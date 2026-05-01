"""
Feature Engineering Module
===========================

Computes the 11 stationary technical features and auxiliary columns that
enter the walk-forward training pipeline.

Role in the pipeline
--------------------
Called by ``main.py`` after data alignment (Step 5) and before triple-barrier
labeling (Step 6).  The output DataFrame is the input to ``TripleBarrierLabeler``,
which appends the ``label`` and ``label_binary`` columns before the full dataset
is saved to HDF5 under the ``engineered_features`` key.

Feature design: why 11 stationary features?
---------------------------------------------
All 11 features are dimensionless (price-scale independent).  Non-stationary
representations — absolute price levels, raw MACD in dollars, raw ATR in dollars —
were excluded because they suffer from covariate shift: a value that was "high"
in 2010 is "normal" in 2024 at a different price level.  The walk-forward scheme
evaluates the model on unseen future data; non-stationary features would silently
degrade generalisation across folds.

The 11 active model features (used by all training scripts):

    Momentum (2):
        ret_5d   — 5-day percentage return; captures short-term trend
        ret_21d  — 21-day percentage return; captures monthly trend

    Volatility (1):
        vol_20d  — EWM std of daily returns (span=20); consistent with the
                   triple-barrier barrier width (k × vol_20d)

    RSI (1):
        rsi_14   — Wilder RSI (14-period EWM, alpha=1/14); dimensionless [0–100]

    MACD (3):
        macd_line, macd_signal, macd_hist — normalised by Close price (% of price)
        so the signal is comparable across price regimes

    Bollinger Bands (2):
        bb_pct   — position within the bands [0=lower, 1=upper]
        bb_width — (upper−lower)/mid; relative band width

    ATR (1):
        atr_14   — Wilder ATR normalised by Close (ATR%); relative volatility

    Volume (1):
        volume_ratio — current volume / 20-day MA; detects unusual activity

Additional computed columns (not model features):
    ret_1d           — daily return; intermediate required for vol_20d
    ret_1d_forward   — next-day return for P&L backtesting (NOT a feature)
    ret_10d_forward  — 10-day forward return; IC target for feature selection
    label_10d_binary — simple 10-day direction sign (legacy, not active label)
    ret_5d_forward   — 5-day forward return (legacy, not used by active pipeline)
    label_5d_binary  — simple 5-day direction sign (legacy, not active label)
    sma_200_dist     — distance from 200-day SMA (computed but not in active X)

Design decisions
----------------
1. **Per-ticker calculation** — every indicator is computed independently for
   each ticker.  Mixing tickers would introduce cross-sectional look-ahead bias
   (e.g. a normalisation that uses future tickers' statistics).

2. **EWM volatility (not rolling std)** — tutor specification.  EWM down-weights
   old observations, making vol_20d more responsive to recent regime changes.
   This is also consistent with the triple-barrier labeling which uses the same
   vol_20d as the barrier width.

3. **MACD normalised by Close** — the raw MACD (EMA_12 − EMA_26) is in dollar
   terms and scales with the price level.  Dividing by Close converts it to a
   percentage, making it comparable across the 2004–2024 price range (~$90–$600).

4. **ATR via Wilder's EWM (alpha=1/n)** — matches the original Wilder (1978)
   definition.  Simple rolling mean would produce a different (slower) signal
   and would be inconsistent with the RSI calculation which also uses Wilder EWM.

5. **bb_pct and bb_width only (no bb_mid/bb_upper/bb_lower)** — absolute band
   levels are non-stationary price levels and would reintroduce the covariate
   shift problem.  Only the two dimensionless derived measures are kept.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from loguru import logger


# The 11 active model features — used by all training scripts to build X.
# Listed here for reference; the canonical source is each training script's
# FEATURE_COLS constant.
ACTIVE_FEATURES = [
    'ret_5d', 'ret_21d',                                    # momentum
    'vol_20d',                                               # volatility
    'rsi_14',                                                # oscillator
    'macd_line', 'macd_signal', 'macd_hist',                 # trend
    'bb_pct', 'bb_width',                                    # bands
    'atr_14',                                                # volatility
    'volume_ratio',                                          # volume
]


class FeatureEngineer:
    """
    Computes technical features for financial time series data.

    Processes each ticker independently (see module docstring — design decision 1).
    Returns the full DataFrame with both model features and auxiliary columns
    (forward returns, legacy labels); the training scripts select only the 11
    active features via their FEATURE_COLS constant.

    Active features (11 stationary, dimensionless):
        ret_5d, ret_21d, vol_20d, rsi_14,
        macd_line, macd_signal, macd_hist,
        bb_pct, bb_width, atr_14, volume_ratio

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        momentum_periods (list): Periods for pct_change returns (default [1, 5, 21]).
        volatility_period (int): EWM span for vol_XXd (default 20).
        rsi_period (int): RSI look-back (default 14).
        macd_fast, macd_slow, macd_signal (int): MACD parameters (12/26/9).
        bollinger_period (int): Bollinger SMA window (default 20).
        bollinger_std (int): Number of std for bands (default 2).
        atr_period (int): ATR look-back (default 14).
        volume_ma_period (int): Volume MA window (default 20).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise FeatureEngineer from the project configuration.

        Args:
            config: Configuration dictionary.  Reads the 'features.technical'
                    section for all indicator parameters.
        """
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
        logger.info(f"Volatility: EWM span={self.volatility_period}")
        logger.info(
            f"RSI: period={self.rsi_period}  |  "
            f"MACD: {self.macd_fast}/{self.macd_slow}/{self.macd_signal} (normalised by Close)  |  "
            f"ATR: period={self.atr_period} (Wilder, normalised by Close)"
        )
        logger.info(
            f"Bollinger: period={self.bollinger_period}, std={self.bollinger_std} "
            f"(bb_pct + bb_width only)  |  Volume MA: {self.volume_ma_period}"
        )

    # ------------------------------------------------------------------
    # Individual indicator calculators (called per ticker)
    # ------------------------------------------------------------------

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum returns, forward returns, and legacy label columns.

        Columns produced and their roles
        ---------------------------------
        ret_1d         (intermediate) — daily return; required by calculate_volatility.
        ret_5d         (feature)      — 5-day momentum; in active FEATURE_COLS.
        ret_21d        (feature)      — 21-day momentum; in active FEATURE_COLS.
        ret_1d_forward (backtesting)  — next-day return for P&L calculation.
                                        MUST NOT be used as a model feature
                                        (would introduce look-ahead bias).
        ret_10d_forward (IC target)   — 10-day forward return used as the
                                        information coefficient (IC) target for
                                        optional feature selection.  Last 10 rows
                                        are NaN (no forward data available).
        ret_5d_forward  (legacy)      — 5-day forward return; computed but not
                                        used by the active pipeline.
        label_5d_binary (legacy)      — sign of ret_5d_forward; not the active
                                        training label (active label = label_binary
                                        from triple-barrier labeling).
        label_10d_binary (legacy)     — sign of ret_10d_forward; not the active
                                        training label.

        Args:
            data: Single-ticker DataFrame with a 'Close' column.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with all return and label columns appended.
        """
        # --- Model features: momentum returns ---
        for period in self.momentum_periods:
            data[f'ret_{period}d'] = data['Close'].pct_change(period)
            logger.debug(f"Calculated ret_{period}d")

        # --- Backtesting artifact: next-day realized return ---
        # ret_1d_forward = tomorrow's return, used to compute strategy P&L.
        # It is shift(-1) of ret_1d — i.e. (Close[t+1] - Close[t]) / Close[t].
        data['ret_1d_forward'] = data['ret_1d'].shift(-1)
        logger.debug("Calculated ret_1d_forward (P&L backtesting only — not a feature)")

        # --- IC target: 10-day forward return ---
        # Used as the IC (Spearman correlation) target during optional feature
        # selection.  Last 10 rows are NaN (no future data).
        data['ret_10d_forward'] = data['Close'].pct_change(10).shift(-10)
        logger.debug("Calculated ret_10d_forward (IC feature selection target)")

        # --- Legacy: 10-day direction sign ---
        # Not the active training label; kept for backward compatibility.
        # Active label = label_binary (from triple-barrier, in triple_barrier.py).
        data['label_10d_binary'] = np.where(
            data['ret_10d_forward'].isna(), np.nan,
            np.where(data['ret_10d_forward'] > 0, 1.0, -1.0)
        )
        logger.debug("Calculated label_10d_binary (legacy — not used by active pipeline)")

        # --- Legacy: 5-day forward return and direction sign ---
        data['ret_5d_forward'] = data['ret_5d'].shift(-5)
        data['label_5d_binary'] = np.where(
            data['ret_5d_forward'].isna(), np.nan,
            np.where(data['ret_5d_forward'] > 0, 1.0, -1.0)
        )
        logger.debug("Calculated ret_5d_forward, label_5d_binary (legacy — not used by active pipeline)")

        return data

    def calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute exponentially-weighted volatility of daily returns (vol_20d).

        Design rationale
        ~~~~~~~~~~~~~~~~
        EWM std with span=20 down-weights old observations, making vol_20d more
        responsive to recent regime changes than a simple rolling standard
        deviation.  Tutor specification: this exact measure is also used as the
        barrier width in triple-barrier labeling (barriers = k × vol_20d), so
        using the same vol_20d in the feature matrix creates a direct relationship
        between the label target and the feature space.

        Requires ``ret_1d`` — must be called after ``calculate_returns``.

        Args:
            data: Single-ticker DataFrame with 'ret_1d' column.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with vol_20d column added.

        Raises:
            ValueError: If 'ret_1d' is absent (wrong call order).
        """
        if 'ret_1d' not in data.columns:
            raise ValueError(
                "'ret_1d' not found.  "
                "calculate_returns must be called before calculate_volatility."
            )

        data[f'vol_{self.volatility_period}d'] = (
            data['ret_1d'].ewm(span=self.volatility_period).std()
        )
        logger.debug(
            f"Calculated vol_{self.volatility_period}d  |  "
            f"EWM(span={self.volatility_period})"
        )

        return data

    def calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI (Relative Strength Index) using Wilder's smoothing.

        Implementation
        ~~~~~~~~~~~~~~
        Wilder's original RSI uses an EMA with alpha = 1/n (not a simple
        rolling mean).  ``ewm(alpha=1/n, adjust=False)`` implements the
        recursive formula:
            avg_gain[t] = alpha × gain[t] + (1 − alpha) × avg_gain[t−1]

        The resulting rsi_14 is bounded [0, 100] and dimensionless — no
        price-scale normalisation is needed.

        Args:
            data: Single-ticker DataFrame with 'Close' column.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with rsi_14 column added.
        """
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).ewm(
            alpha=1 / self.rsi_period, adjust=False
        ).mean()
        loss = (-delta.clip(upper=0)).ewm(
            alpha=1 / self.rsi_period, adjust=False
        ).mean()

        data[f'rsi_{self.rsi_period}'] = 100 - (100 / (1 + gain / loss))
        logger.debug(f"Calculated rsi_{self.rsi_period}  |  Wilder EWM(alpha=1/{self.rsi_period})")

        return data

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MACD (12/26/9) normalised by Close price.

        Price normalisation rationale
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The raw MACD line (EMA_12 − EMA_26) is denominated in dollars and scales
        with the price level: a reading of 2.0 meant something different when
        SPY was at $100 (2004) vs $450 (2024).  Dividing by Close converts all
        three MACD values to a percentage of price, making the signal comparable
        across the full 2004–2024 range and preventing covariate shift between
        walk-forward folds.

            macd_line   = (EMA_12 − EMA_26) / Close × 100   [% of price]
            macd_signal = EMA_9(macd_line)                   [% of price]
            macd_hist   = macd_line − macd_signal            [% of price]

        Args:
            data: Single-ticker DataFrame with 'Close' column.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with macd_line, macd_signal, macd_hist columns added.
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
        Compute stationary Bollinger Band features (bb_pct, bb_width).

        Only two dimensionless features are added to the DataFrame.  The
        intermediate band levels (bb_mid, bb_upper, bb_lower) are absolute
        price values and would be non-stationary if included.  They are
        computed as local variables and discarded after deriving the two
        relative measures:

            bb_pct   = (Close − bb_lower) / (bb_upper − bb_lower)
                       Price position within the bands, bounded ~[0, 1].
                       Values near 0 indicate close to the lower band (oversold);
                       values near 1 indicate close to the upper band (overbought).

            bb_width = (bb_upper − bb_lower) / bb_mid
                       Relative band width normalised by the 20-day SMA.
                       Captures volatility contraction (squeeze) and expansion
                       without price-level dependence.

        Args:
            data: Single-ticker DataFrame with 'Close' column.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with bb_pct and bb_width columns added.
        """
        # Intermediate values — local variables only, never added to DataFrame
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
        Compute ATR (Average True Range) via Wilder's smoothing, normalised by
        Close price (ATR%).

        Wilder's smoothing
        ~~~~~~~~~~~~~~~~~~
        ``ewm(alpha=1/n, adjust=False)`` implements Wilder's recursive formula:
            ATR[t] = alpha × TrueRange[t] + (1 − alpha) × ATR[t−1]
        This differs from a simple rolling mean and matches the original
        definition in Wilder (1978).  It is also consistent with the RSI
        calculation which uses the same alpha convention.

        Price normalisation
        ~~~~~~~~~~~~~~~~~~~
        Raw ATR is in dollars.  Dividing by Close gives ATR% — a relative
        volatility measure comparable across the 2004–2024 price range.

            atr_14 = EWM_α(1/14)(TrueRange) / Close × 100   [% of price]

        Args:
            data: Single-ticker DataFrame with 'High', 'Low', 'Close' columns.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with atr_14 column added (% of Close price).
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
        Compute volume ratio: current volume divided by its 20-day rolling mean.

        A ratio > 1 indicates above-average trading activity (breakout
        confirmation, panic selling).  Dimensionless by construction — no
        price-scale normalisation needed.

        Args:
            data: Single-ticker DataFrame with 'Volume' column.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with volume_ratio column added.
        """
        volume_ma = data['Volume'].rolling(window=self.volume_ma_period).mean()
        data['volume_ratio'] = data['Volume'] / volume_ma

        logger.debug(
            f"Calculated volume_ratio  |  MA period={self.volume_ma_period}"
        )

        return data

    def calculate_sma_200_dist(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute distance from the 200-day Simple Moving Average (SMA).

        Positive = price above the long-term trend (bullish).
        Negative = price below the long-term trend (bearish).

        Note: This column is computed and stored in HDF5 but is NOT included
        in the active 11-feature set (ACTIVE_FEATURES).  It is available for
        exploratory analysis but excluded from model training to keep the
        feature set parsimonious and avoid multicollinearity with the momentum
        features (ret_5d, ret_21d).

        Args:
            data: Single-ticker DataFrame with 'Close' column.
                  Mutated in-place — caller must pass an owned copy.

        Returns:
            Same DataFrame with sma_200_dist column added.
        """
        sma_200 = data['Close'].rolling(window=200).mean()
        data['sma_200_dist'] = (data['Close'] - sma_200) / sma_200

        logger.debug("Calculated sma_200_dist (not in active feature set)")

        return data

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def engineer_features_for_ticker(self, ticker_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature calculations to a single ticker's OHLCV data.

        Call order is fixed: ``calculate_returns`` must precede
        ``calculate_volatility`` because vol_20d depends on ret_1d.

        Args:
            ticker_data: Single-ticker DataFrame with OHLCV columns.

        Returns:
            DataFrame with all feature and auxiliary columns appended.
        """
        result = ticker_data.copy()

        result = self.calculate_returns(result)       # ret_5d, ret_21d (features); ret_1d (intermediate)
        result = self.calculate_volatility(result)    # vol_20d — depends on ret_1d
        result = self.calculate_rsi(result)           # rsi_14
        result = self.calculate_macd(result)          # macd_line, macd_signal, macd_hist
        result = self.calculate_bollinger_bands(result)  # bb_pct, bb_width
        result = self.calculate_atr(result)           # atr_14
        result = self.calculate_volume_ratio(result)  # volume_ratio
        result = self.calculate_sma_200_dist(result)  # sma_200_dist (not in active feature set)

        return result

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to all tickers in the aligned dataset.

        Each ticker is processed independently (design decision 1) to prevent
        cross-sectional look-ahead bias.  Results are concatenated and sorted
        by (ticker, date) to restore the MultiIndex structure expected by
        ``TripleBarrierLabeler`` and all downstream training scripts.

        Args:
            data: MultiIndex (ticker, date) DataFrame with OHLCV + VIX columns.

        Returns:
            MultiIndex (ticker, date) DataFrame with all technical features and
            auxiliary columns appended to the original OHLCV + VIX columns.
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
        logger.info(
            f"Active features ({len(ACTIVE_FEATURES)}): {ACTIVE_FEATURES}"
        )
        logger.info(
            f"All new columns ({len(new_cols)}): {new_cols}"
        )
        logger.info(f"Output shape: {result.shape}")
        logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Inspection utility
    # ------------------------------------------------------------------

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a quality summary for an engineered DataFrame.

        Classifies every column into one of four categories:
        - OHLCV: raw price/volume columns from yfinance.
        - FRED: macro series and their derived columns (read from config).
        - Non-feature auxiliaries: forward returns, legacy labels, intermediates.
        - Model features: columns that enter the training feature matrix X.

        Args:
            data: DataFrame returned by engineer_features() (before or after
                  triple-barrier labeling).

        Returns:
            Dictionary with counts, column lists, and missing value totals.
        """
        ohlcv_cols = {'Close', 'High', 'Low', 'Open', 'Volume'}

        # Columns produced by this module that are NOT model features
        auxiliary_cols = {
            'ret_1d',           # intermediate — only needed for vol_20d
            'ret_1d_forward',   # backtesting P&L
            'ret_5d_forward',   # legacy
            'label_5d_binary',  # legacy
            'ret_10d_forward',  # IC target
            'label_10d_binary', # legacy
            'sma_200_dist',     # exploratory — not in active feature set
        }

        # Triple-barrier label columns (added by TripleBarrierLabeler, not here)
        label_cols = {'label', 'label_binary'}

        # FRED series and derived columns — read dynamically from config
        fred_base = list(self.config.get('fred_series', {}).keys())
        fred_cols = set()
        for base in fred_base:
            fred_cols.add(base)
            fred_cols.add(f'{base}_diff')
            fred_cols.add(f'{base}_chg')

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
