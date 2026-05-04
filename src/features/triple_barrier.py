"""
Triple Barrier Labeling — López de Prado (AFML, 2018, Chapter 3).

Called by main.py (Step 6) after feature engineering. Appends label,
label_binary, and days_to_barrier to the DataFrame persisted in HDF5.
All training scripts use label_binary as the target y.

For each bar, the first of three barriers touched within max_holding_period days:
    Upper (take profit): Close[t] × (1 + k × vol_20d[t])
    Lower (stop loss):   Close[t] × (1 − k × vol_20d[t])
    Time barrier:        max_holding_period trading days  → label = 0

Binary collapse (tutor specification):
    label = +1 → label_binary = +1
    label = -1 → label_binary = -1
    label =  0 → label_binary = -1   (time barrier treated as loss)

Parameters (config): k=1.0 (vol_multiplier), max_holding_period=8 days.
NaN vol_20d rows receive NaN labels and are dropped by training scripts.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class TripleBarrierLabeler:
    """
    Labels each bar by the first barrier touched within max_holding_period days.
    Barrier width = max(k × vol_20d, min_ret) — dynamic, volatility-scaled.
    Output: label (ternary ±1/0), label_binary (±1), days_to_barrier.
    """

    def __init__(self, config: Dict[str, Any]):
        """Reads max_holding_period, vol_multiplier, min_ret from config['features']['triple_barrier']."""
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
        Scalar reference implementation for debugging. Returns +1, -1, or 0.
        Production pipeline uses label_ticker_data (NumPy vectorized).
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
        Sentinel n represents "barrier never crossed" — always > any valid crossing index.
        Returns (DataFrame with label/label_binary/days_to_barrier, stats dict).
        Raises ValueError if 'vol_20d' is absent.
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
        Apply triple-barrier labeling to all tickers independently.
        Logs per-ticker and overall label distribution. Warns if time barrier > 40%.
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
        """Return overall and per-ticker ternary/binary label distribution."""
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
