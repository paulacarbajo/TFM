"""
Triple Barrier Labeling Module
================================

Generates trade labels using the triple-barrier method of López de Prado
(Advances in Financial Machine Learning, 2018, Chapter 3).

Role in the pipeline
--------------------
Called by ``main.py`` as Step 6, immediately after feature engineering.
Appends three columns — ``label``, ``label_binary``, ``days_to_barrier`` —
to the DataFrame that is then persisted to HDF5 under ``engineered_features``.
All training scripts use ``label_binary`` as the target variable **y**.

Why triple barrier instead of a fixed-horizon direction label?
---------------------------------------------------------------
A simple label like ``label_10d_binary = sign(ret_10d_forward)`` evaluates
the trade outcome at a fixed horizon (day 10) regardless of what happened in
between.  Problems:

1. **Path blindness** — a trade that returned −20 % on day 9 and recovered to
   +0.1 % on day 10 would be labeled +1 (profitable), masking the drawdown.

2. **No risk control** — a real systematic strategy always has a stop-loss.
   Ignoring it produces over-optimistic labels that cannot be replicated in
   live trading.

3. **Arbitrary horizon** — selecting 10 days is arbitrary and bakes in an
   assumption about the regime (trending vs mean-reverting).

The triple barrier evaluates each observation against two dynamic barriers
(based on current realized volatility) and a time barrier:
- **Upper barrier** (take profit) = entry price × (1 + k × vol_20d)
- **Lower barrier** (stop loss)   = entry price × (1 − k × vol_20d)
- **Time barrier**                = max_holding_period trading days

The label is assigned to whichever is touched first.

Parameter choices (from config)
--------------------------------
- **k = 1.0 (vol_multiplier)** — barriers set at ±1 standard deviation of
  realized volatility.  Tutor specification.  A smaller k produces narrower
  barriers, more stop-losses, and a noisier label.  A larger k produces
  labels that fire less frequently.
- **max_holding_period = 8 days** — upper bound on trade duration.  Short
  enough to stay in the tactical horizon of the technical indicators used
  (RSI-14, MACD-12/26) while long enough for vol_20d to produce meaningful
  barriers.

Binary label design decision
------------------------------
``label_binary`` collapses the ternary label:
    label =  1  →  label_binary =  1   (take profit)
    label = -1  →  label_binary = -1   (stop loss)
    label =  0  →  label_binary = -1   (time barrier → treated as loss)

Rationale: collapsing time barrier into -1 reflects a conservative assumption:
a trade that failed to reach the take-profit within the holding period did not
deliver the expected alpha.  Treating it as −1 rather than 0 slightly increases
the negative class weight, which is appropriate given that a real strategy
incurs opportunity cost (capital tied up, transaction costs) even when the
time barrier fires without a loss.  Tutor specification.

NaN handling
-------------
The first row of vol_20d is NaN (EWM std of a single observation is
undefined).  Rows where vol_20d is NaN receive ``label = NaN`` and
``label_binary = NaN`` — they are excluded from training automatically by
pandas ``dropna()`` in the training scripts.

Near-end rows (last ``max_holding_period`` rows) receive a label based on a
truncated future window; these tend to be labeled 0 (time barrier) more
frequently than earlier rows because fewer future prices are available.
Training scripts that use a strict train/validation split are unaffected
since these rows fall inside the validation period.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class TripleBarrierLabeler:
    """
    Labels financial time series using the triple-barrier method.

    For each observation at time t, three barriers are defined:
    - Upper barrier: Close[t] × (1 + k × vol_20d[t])
    - Lower barrier: Close[t] × (1 − k × vol_20d[t])
    - Time barrier: max_holding_period trading days

    The barrier touched first determines the ternary label:
        label = +1   upper touched first  (take profit)
        label = -1   lower touched first  (stop loss)
        label =  0   time expires first   (no clear signal)

    The binary label (used as model target y):
        label_binary = +1   if label =  1
        label_binary = -1   if label = -1 or 0

    See module docstring for the full rationale.

    Attributes:
        config (Dict[str, Any]):  Configuration dictionary.
        max_holding_period (int): Time barrier in trading days (default 8).
        vol_multiplier (float):   Barrier width multiplier k (default 1.0).
        min_ret (float):          Minimum absolute barrier width (default 0.0).
                                  Ensures a non-zero barrier even when vol ≈ 0.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise TripleBarrierLabeler from the project configuration.

        Args:
            config: Configuration dictionary.  Reads 'features.triple_barrier'
                    for max_holding_period, vol_multiplier, and min_ret.
        """
        self.config = config
        features_config = config.get('features', {})
        tb_config = features_config.get('triple_barrier', {})

        self.max_holding_period = tb_config.get('max_holding_period', 8)
        self.vol_multiplier = tb_config.get('vol_multiplier', 1.0)
        self.min_ret = tb_config.get('min_ret', 0.0)

        logger.info("TripleBarrierLabeler initialised")
        logger.info(
            f"Parameters: k={self.vol_multiplier}  |  "
            f"max_holding={self.max_holding_period}d  |  "
            f"min_ret={self.min_ret}"
        )
        logger.info(
            "Barrier width = max(k × vol_20d, min_ret)  —  "
            "dynamic, volatility-scaled barriers"
        )

    # ------------------------------------------------------------------
    # Core labeling logic
    # ------------------------------------------------------------------

    def get_barrier_for_observation(
        self,
        close_prices: pd.Series,
        current_idx: int,
        current_price: float,
        volatility: float
    ) -> int:
        """
        Determine the label for a single observation (scalar, non-vectorized).

        This is a reference implementation for debugging and unit testing.
        The main pipeline uses ``label_ticker_data`` (NumPy vectorized) for
        performance.  Both functions implement identical logic and must return
        the same ternary label for any valid input.

        Args:
            close_prices: Full Close price series for the ticker.
            current_idx:  Integer position of the observation in close_prices.
            current_price: Close price at current_idx.
            volatility:   vol_20d value at current_idx (EWM std of ret_1d).

        Returns:
            +1 (upper barrier), −1 (lower barrier), or 0 (time barrier).
        """
        threshold = max(volatility * self.vol_multiplier, self.min_ret)
        upper_barrier = current_price * (1 + threshold)
        lower_barrier = current_price * (1 - threshold)

        # +1 so the slice exactly covers max_holding_period prices
        end_idx = min(current_idx + self.max_holding_period + 1, len(close_prices))
        future_prices = close_prices.iloc[current_idx + 1:end_idx]

        if len(future_prices) == 0:
            return 0

        for price in future_prices:
            if price >= upper_barrier:
                return 1
            elif price <= lower_barrier:
                return -1

        return 0

    def label_ticker_data(
        self, ticker_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply triple-barrier labeling to a single ticker (NumPy vectorized).

        Vectorization approach
        ~~~~~~~~~~~~~~~~~~~~~~
        For each bar i the future price window is ``close[i+1 : i+H+1]`` where
        H = max_holding_period.  ``np.where(future >= upper)`` and
        ``np.where(future <= lower)`` return arrays of crossing indices.
        The sentinel value ``n`` (total series length) is used to represent
        "never crossed" — it is always larger than any valid crossing index
        (which is bounded by H ≤ 8 < n), so comparisons with ``<`` work
        correctly without a special-case branch.

        Binary label collapse
        ~~~~~~~~~~~~~~~~~~~~~
        ``label_binary`` is derived after the full ternary array is computed:
            label_binary = np.where(labels == 1, 1, -1)
        NaN propagation is applied in a second pass so that rows with NaN vol
        receive NaN in both label and label_binary.

        Args:
            ticker_data: Single-ticker DataFrame containing 'Close' and 'vol_20d'.

        Returns:
            Tuple of:
            - DataFrame with 'label', 'label_binary', 'days_to_barrier' added.
            - Stats dict: take_profit, stop_loss, time_barrier, total_valid,
              avg_days_to_barrier.

        Raises:
            ValueError: If 'vol_20d' is absent (feature engineering not run).
        """
        result = ticker_data.copy()

        if 'vol_20d' not in result.columns:
            raise ValueError(
                "'vol_20d' column not found.  "
                "Run feature engineering before triple-barrier labeling.  "
                f"Available columns: {result.columns.tolist()}"
            )

        close = result['Close'].values
        vols = result['vol_20d'].values
        n = len(close)

        labels          = np.zeros(n)
        days_to_barrier = np.zeros(n)

        take_profit_count  = 0
        stop_loss_count    = 0
        time_barrier_count = 0

        for i in range(n):
            if np.isnan(vols[i]) or np.isnan(close[i]):
                labels[i]          = np.nan
                days_to_barrier[i] = np.nan
                continue

            threshold = max(vols[i] * self.vol_multiplier, self.min_ret)
            upper = close[i] * (1 + threshold)
            lower = close[i] * (1 - threshold)

            end    = min(i + self.max_holding_period + 1, n)
            future = close[i + 1:end]

            if len(future) == 0:
                # Last bar in the series — no future prices available
                labels[i]          = 0
                days_to_barrier[i] = 0
                time_barrier_count += 1
                continue

            up_cross = np.where(future >= upper)[0]
            dn_cross = np.where(future <= lower)[0]

            # n is a safe "never" sentinel: crossing indices are < H ≤ 8 < n
            first_up = up_cross[0] if len(up_cross) > 0 else n
            first_dn = dn_cross[0] if len(dn_cross) > 0 else n

            if first_up == n and first_dn == n:
                # Neither barrier touched within the holding window
                labels[i]          = 0
                days_to_barrier[i] = len(future)
                time_barrier_count += 1
            elif first_up <= first_dn:
                # Take profit touched first (tie goes to take profit)
                labels[i]          = 1
                days_to_barrier[i] = first_up + 1
                take_profit_count  += 1
            else:
                # Stop loss touched first
                labels[i]          = -1
                days_to_barrier[i] = first_dn + 1
                stop_loss_count    += 1

        # --- Ternary label ---
        result['label'] = labels

        # --- Binary label: collapse time barrier (0) into loss (-1) ---
        # Two-step to preserve NaN: first collapse, then re-inject NaN.
        label_binary = np.where(labels == 1, 1, -1)
        label_binary = np.where(np.isnan(labels), np.nan, label_binary)
        result['label_binary']    = label_binary
        result['days_to_barrier'] = days_to_barrier

        valid_mask  = ~np.isnan(labels)
        total_valid = int(valid_mask.sum())

        stats: Dict[str, Any] = {
            'take_profit':         take_profit_count,
            'stop_loss':           stop_loss_count,
            'time_barrier':        time_barrier_count,
            'total_valid':         total_valid,
            'avg_days_to_barrier': float(np.nanmean(days_to_barrier)) if total_valid > 0 else 0.0,
        }

        return result, stats

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple-barrier labeling to all tickers in the dataset.

        Each ticker is labeled independently (no look-ahead across tickers).
        Aggregated statistics are logged so that parameter quality can be
        assessed without re-running the full pipeline.

        Time-barrier warning threshold
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        If the time barrier fires in >40 % of cases the current parameters
        may be too tight (k too large, max_holding too short) or the regime
        is strongly mean-reverting.  A warning is emitted per ticker and
        for the overall dataset.

        Args:
            data: MultiIndex (ticker, date) DataFrame with 'Close' and
                  'vol_20d' columns (output of FeatureEngineer).

        Returns:
            MultiIndex (ticker, date) DataFrame with 'label', 'label_binary',
            and 'days_to_barrier' columns appended.
        """
        logger.info("=" * 60)
        logger.info("TRIPLE BARRIER LABELING")
        logger.info("=" * 60)
        logger.info(f"Input shape: {data.shape}")
        logger.info(
            f"Parameters: k={self.vol_multiplier}  |  "
            f"max_holding={self.max_holding_period}d  |  "
            f"min_ret={self.min_ret}"
        )

        tickers = data.index.get_level_values('ticker').unique()
        logger.info(f"Processing {len(tickers)} ticker(s): {tickers.tolist()}")

        ticker_results       = []
        total_take_profit    = 0
        total_stop_loss      = 0
        total_time_barrier   = 0
        total_valid          = 0
        all_days_to_barrier  = []

        for ticker in tickers:
            logger.info(f"  Labeling {ticker}")

            ticker_data   = data.xs(ticker, level='ticker')
            ticker_labeled, stats = self.label_ticker_data(ticker_data)

            tp    = stats['take_profit']
            sl    = stats['stop_loss']
            tb    = stats['time_barrier']
            valid = stats['total_valid']

            total_take_profit  += tp
            total_stop_loss    += sl
            total_time_barrier += tb
            total_valid        += valid

            if valid > 0:
                logger.info(
                    f"  {ticker}  |  "
                    f"take_profit={tp} ({tp/valid*100:.1f}%)  |  "
                    f"stop_loss={sl} ({sl/valid*100:.1f}%)  |  "
                    f"time_barrier={tb} ({tb/valid*100:.1f}%)  |  "
                    f"avg_days={stats['avg_days_to_barrier']:.2f}"
                )
                all_days_to_barrier.extend(
                    ticker_labeled['days_to_barrier'].dropna().tolist()
                )
                if tb / valid > 0.4:
                    logger.warning(
                        f"  {ticker}: time barrier {tb/valid*100:.1f}% > 40% — "
                        f"consider increasing k or max_holding_period"
                    )

            ticker_labeled['ticker'] = ticker
            ticker_labeled = ticker_labeled.reset_index().set_index(['ticker', 'date'])
            ticker_results.append(ticker_labeled)
            logger.success(f"  {ticker} done")

        result = pd.concat(ticker_results).sort_index()

        # --- Overall statistics ---
        logger.info("=" * 60)
        logger.info("OVERALL STATISTICS")
        logger.info("=" * 60)

        if total_valid > 0:
            tp_pct  = total_take_profit  / total_valid * 100
            sl_pct  = total_stop_loss    / total_valid * 100
            tb_pct  = total_time_barrier / total_valid * 100
            avg_days = np.mean(all_days_to_barrier) if all_days_to_barrier else 0.0

            logger.info(f"Total valid observations: {total_valid}")
            logger.info(f"Take profit  (label= 1): {total_take_profit:6d} ({tp_pct:5.2f}%)")
            logger.info(f"Stop loss    (label=-1): {total_stop_loss:6d}   ({sl_pct:5.2f}%)")
            logger.info(f"Time barrier (label= 0): {total_time_barrier:6d} ({tb_pct:5.2f}%)")
            logger.info(f"Average days to barrier: {avg_days:.2f}")

            logger.info("Binary label distribution (model target y):")
            binary_counts = result['label_binary'].value_counts().sort_index()
            valid_binary  = result['label_binary'].notna().sum()
            for val, count in binary_counts.items():
                if pd.notna(val):
                    pct = count / valid_binary * 100
                    logger.info(f"  label_binary {int(val):+d}: {count:6d} ({pct:5.2f}%)")

            if tb_pct > 40:
                logger.warning(
                    f"Time barrier {tb_pct:.1f}% > 40% overall — "
                    f"k={self.vol_multiplier}, max_holding={self.max_holding_period}"
                )
            else:
                logger.success(
                    f"Time barrier {tb_pct:.1f}% — within acceptable range (<40%)"
                )

        logger.info("=" * 60)
        logger.success("TRIPLE BARRIER LABELING COMPLETE")
        logger.info(f"Output shape: {result.shape}")
        logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Inspection utility
    # ------------------------------------------------------------------

    def get_label_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a label distribution summary for an already-labeled DataFrame.

        Args:
            data: MultiIndex (ticker, date) DataFrame with 'label' and
                  'label_binary' columns.

        Returns:
            Dictionary with overall and per-ticker ternary/binary distributions.
        """
        summary: Dict[str, Any] = {
            'total_observations':       len(data),
            'missing_labels':           int(data['label'].isna().sum()),
            'ternary_label_distribution': data['label'].value_counts().to_dict(),
            'binary_label_distribution':  data['label_binary'].value_counts().to_dict(),
            'ternary_label_percentages': (
                data['label'].value_counts() / len(data) * 100
            ).to_dict(),
            'per_ticker': {},
        }

        for ticker in data.index.get_level_values('ticker').unique():
            td = data.xs(ticker, level='ticker')
            summary['per_ticker'][ticker] = {
                'total':                len(td),
                'ternary_distribution': td['label'].value_counts().to_dict(),
                'binary_distribution':  td['label_binary'].value_counts().to_dict(),
            }

        return summary
