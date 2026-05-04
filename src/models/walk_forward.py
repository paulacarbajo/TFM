"""
Walk-Forward Cross-Validation Module
======================================

Implements rolling-window walk-forward CV for financial time series,
enforcing strict no-look-ahead bias throughout.

Rolling window schema (3-year train / 1-year val, config_2010.yaml)
---------------------------------------------------------------------
    Fold 1: train 2010–2013, val 2013–2014
    Fold 2: train 2011–2014, val 2014–2015
    Fold 3: train 2012–2015, val 2015–2016
    Fold 4: train 2013–2016, val 2016–2017
    Fold 5: train 2014–2017, val 2017–2018
    Fold 6: train 2015–2018, val 2018–2019
    Fold 7: train 2016–2019, val 2019–2020
    OOS:    2020–2024  (never seen during training)

No-look-ahead guarantees
-------------------------
1. Date filtering: end date is always exclusive — train and val windows
   are strictly non-overlapping.
2. IC feature selection: Spearman correlation computed on the training
   slice only; the selected feature set is then applied to validation.
3. Regime detection (RegimeDetector): fitted on training data only,
   applied via transform to validation (handled in calling scripts).
4. Expanding window: legacy branch — not used in production (raises
   NotImplementedError if invoked).
"""

from typing import Dict, Any, List, Tuple, Generator

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from loguru import logger


# Canonical 10-feature set used by all training scripts.
# Order is fixed here — get_feature_names preserves this order regardless
# of column order in the input DataFrame.
TECHNICAL_FEATURES = [
    'ret_5d', 'ret_21d',                                    # momentum
    'vol_rel', 'atr_14',                                    # volatility (vol_rel = vol_20d/vol_60d)
    'rsi_14',                                               # oscillator
    'macd_line', 'macd_signal',                             # trend (macd_hist excluded: identity of line - signal)
    'bb_pct', 'bb_width',                                   # bands
    'volume_direction',                                     # directional volume (volume_ratio × sign(ret_1d))
    # vol_20d excluded: barrier width in triple-barrier → mechanical correlation with label_binary
]


class WalkForwardCV:
    """
    Rolling-window walk-forward CV. Fixed training window slides forward one
    year per fold. Optional IC feature selection on training slice only.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        models_config = config.get('models', {})
        wf_config     = models_config.get('walk_forward', {})

        self.train_start = pd.to_datetime(wf_config.get('train_start', '2008-01-01'))
        self.train_end   = pd.to_datetime(wf_config.get('train_end',   '2020-01-01'))
        self.test_start  = pd.to_datetime(wf_config.get('test_start',  '2020-01-01'))
        self.test_end    = pd.to_datetime(wf_config.get('test_end',    '2024-12-31'))

        self.window_type        = wf_config.get('window_type', 'rolling')
        self.train_window_years = wf_config.get('train_window_years', 3)
        self.val_window_years   = wf_config.get('val_window_years', 1)

        self.exclude_from_features = models_config.get('exclude_from_features', [
            'Close', 'High', 'Low', 'Open', 'Volume',
            'label', 'label_binary',
        ])

        self.label_column    = wf_config.get('label_column', 'label_binary')
        self.ic_threshold    = wf_config.get('ic_threshold', 0.0)
        self.ic_min_features = wf_config.get('ic_min_features', 6)

        # Purge: remove last max_holding_period trading days from training fold
        # to prevent label overlap with the validation period (López de Prado, AFML ch.7).
        self.purge_days = (
            config.get('features', {})
                  .get('triple_barrier', {})
                  .get('max_holding_period', 0)
        )

        logger.info("WalkForwardCV initialised")
        logger.info(
            f"Window: {self.window_type}  |  "
            f"train={self.train_window_years}yr  |  "
            f"val={self.val_window_years}yr"
        )
        logger.info(
            f"WF period: {self.train_start.date()} → {self.train_end.date()}  |  "
            f"OOS: {self.test_start.date()} → {self.test_end.date()}"
        )
        logger.info(
            f"Label: {self.label_column}  |  "
            f"IC threshold: {self.ic_threshold}  |  "
            f"IC min features: {self.ic_min_features}  |  "
            f"Purge days: {self.purge_days}"
        )

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def get_feature_names(self, data: pd.DataFrame) -> List[str]:
        """
        Return active feature columns in canonical TECHNICAL_FEATURES order.
        Iterates TECHNICAL_FEATURES (not data.columns) for stable order across folds.
        Excluded: vix (GMM only), sma_200_dist, ret_1d (intermediate for vol_20d/volume_direction).
        """
        # Canonical order: iterate TECHNICAL_FEATURES, not data.columns
        feature_columns = [col for col in TECHNICAL_FEATURES if col in data.columns]

        missing = [col for col in TECHNICAL_FEATURES if col not in data.columns]
        if missing:
            logger.warning(f"Features missing from data: {missing}")

        logger.debug(f"Active features ({len(feature_columns)}): {feature_columns}")
        return feature_columns

    def _select_features_by_ic(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> List[str]:
        """
        Filter features by Spearman |IC| vs label_binary on training data only.
        ic_threshold=0.01 (lower than typical 0.02 because binary target yields lower IC).
        Falls back to top ic_min_features if fewer than ic_min_features pass.
        """
        target_col = 'label_binary'

        if target_col not in data.columns:
            logger.warning(
                f"'{target_col}' not found in data — IC selection skipped"
            )
            return feature_columns

        mask  = data[target_col].notna() & data[feature_columns].notna().all(axis=1)
        valid = data[mask]

        if len(valid) < 50:
            logger.warning(
                f"Too few valid rows ({len(valid)}) for IC selection — skipped"
            )
            return feature_columns

        target    = valid[target_col]
        ic_scores = {}
        for feat in feature_columns:
            corr, _ = spearmanr(valid[feat], target)
            ic_scores[feat] = abs(corr) if not np.isnan(corr) else 0.0

        selected = [f for f in feature_columns if ic_scores[f] >= self.ic_threshold]

        if len(selected) < self.ic_min_features:
            selected = sorted(
                feature_columns, key=lambda f: ic_scores[f], reverse=True
            )[:self.ic_min_features]

        logger.info(
            f"IC selection: {len(selected)}/{len(feature_columns)} features kept "
            f"(threshold={self.ic_threshold})"
        )
        for feat in sorted(feature_columns, key=lambda f: ic_scores[f], reverse=True):
            marker = "OK" if feat in selected else "  "
            logger.info(f"  [{marker}] {feat}: IC={ic_scores[feat]:.4f}")

        return selected

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    def _filter_by_date(
        self,
        data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return rows where start_date <= date < end_date (end exclusive, no overlap)."""
        dates = data.index.get_level_values('date')
        return data[(dates >= start_date) & (dates < end_date)].copy()

    def _purge_train(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the last purge_days trading days from the training fold.
        Prevents label overlap with the validation period — a label computed at
        t uses prices up to t + max_holding_period, which may fall inside val.
        López de Prado, AFML ch.7.
        """
        if self.purge_days <= 0:
            return train_data

        dates = train_data.index.get_level_values('date')
        unique_dates = sorted(dates.unique())

        if len(unique_dates) <= self.purge_days:
            logger.warning(
                f"Purge days ({self.purge_days}) >= training dates "
                f"({len(unique_dates)}) — skipping purge"
            )
            return train_data

        cutoff = unique_dates[-self.purge_days]
        purged = train_data[dates < cutoff]
        logger.debug(
            f"Purged last {self.purge_days} trading days  |  "
            f"rows: {len(train_data)} → {len(purged)}"
        )
        return purged

    def _get_fold_dates(self) -> List[Dict[str, pd.Timestamp]]:
        """Compute train/val date ranges for each fold. Only 'rolling' window is supported."""
        if self.window_type != 'rolling':
            raise NotImplementedError(
                f"window_type='{self.window_type}' is not supported.  "
                "Only 'rolling' is validated and used in production.  "
                "The expanding window branch was removed because "
                "self.n_splits was never set in __init__."
            )

        folds: List[Dict[str, pd.Timestamp]] = []
        fold_num            = 1
        current_train_start = self.train_start

        while True:
            train_end = current_train_start + pd.DateOffset(years=self.train_window_years)
            val_start = train_end
            val_end   = val_start + pd.DateOffset(years=self.val_window_years)

            if val_end > self.train_end:
                break

            folds.append({
                'fold':        fold_num,
                'train_start': current_train_start,
                'train_end':   train_end,
                'val_start':   val_start,
                'val_end':     val_end,
            })

            current_train_start += pd.DateOffset(years=self.val_window_years)
            fold_num += 1

        return folds

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_xy(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract X and y, dropping NaN rows (EWM warmup + near-end label NaNs)."""
        X = data[feature_columns]
        y = data[self.label_column]

        valid_mask = y.notna() & X.notna().all(axis=1)
        return X[valid_mask].copy(), y[valid_mask].copy()

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def split(
        self,
        data: pd.DataFrame,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield one dict per fold with keys:
            fold_number, train_start, train_end, val_start, val_end,
            X_train, y_train, X_val, y_val, train_dates, val_dates,
            feature_names, train_data_full, val_data_full.
        train_data_full/val_data_full include vix and ret_1d for RegimeDetector.
        """
        logger.info("=" * 80)
        logger.info("WALK-FORWARD CROSS-VALIDATION")
        logger.info("=" * 80)

        base_features = self.get_feature_names(data)
        logger.info(f"Base features ({len(base_features)}): {base_features}")

        fold_dates  = self._get_fold_dates()
        total_folds = len(fold_dates)
        logger.info(f"Total folds: {total_folds}")

        for fold_info in fold_dates:
            fold_num = fold_info['fold']

            logger.info("")
            logger.info(f"FOLD {fold_num}/{total_folds}  "
                        f"train {fold_info['train_start'].date()} → "
                        f"{fold_info['train_end'].date()}  |  "
                        f"val {fold_info['val_start'].date()} → "
                        f"{fold_info['val_end'].date()}")

            train_data = self._filter_by_date(
                data, fold_info['train_start'], fold_info['train_end']
            )
            train_data = self._purge_train(train_data)
            val_data = self._filter_by_date(
                data, fold_info['val_start'], fold_info['val_end']
            )

            # IC feature selection on training slice only (no look-ahead)
            feature_columns = (
                self._select_features_by_ic(train_data, base_features)
                if self.ic_threshold > 0.0
                else base_features
            )

            X_train, y_train = self._prepare_xy(train_data, feature_columns)
            X_val,   y_val   = self._prepare_xy(val_data,   feature_columns)

            logger.info(
                f"  Train: {len(X_train)} rows  |  "
                f"Val: {len(X_val)} rows  |  "
                f"Features: {len(feature_columns)}"
            )

            # Label distribution (quick sanity check)
            for split_name, y in [('train', y_train), ('val', y_val)]:
                if len(y) > 0:
                    dist = y.value_counts().sort_index()
                    parts = "  ".join(
                        f"label {int(lbl):+d}: {cnt} ({cnt/len(y)*100:.1f}%)"
                        for lbl, cnt in dist.items()
                    )
                    logger.info(f"  {split_name} labels — {parts}")

            yield {
                'fold_number':    fold_num,
                'train_start':    fold_info['train_start'],
                'train_end':      fold_info['train_end'],
                'val_start':      fold_info['val_start'],
                'val_end':        fold_info['val_end'],
                'X_train':        X_train,
                'y_train':        y_train,
                'X_val':          X_val,
                'y_val':          y_val,
                'train_dates':    X_train.index.get_level_values('date'),
                'val_dates':      X_val.index.get_level_values('date'),
                'feature_names':  feature_columns,
                'train_data_full': train_data,   # includes vix, ret_1d for RegimeDetector
                'val_data_full':   val_data,
            }

        logger.info("")
        logger.info("=" * 80)
        logger.success(f"WALK-FORWARD CV COMPLETE  |  {total_folds} folds")
        logger.info("=" * 80)

    def get_oos_data(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Return (X_test, y_test, feature_names) for the held-out OOS period."""
        logger.info("=" * 80)
        logger.info(
            f"OOS TEST DATA  |  "
            f"{self.test_start.date()} → {self.test_end.date()}"
        )

        feature_columns = self.get_feature_names(data)
        test_data       = self._filter_by_date(data, self.test_start, self.test_end)
        X_test, y_test  = self._prepare_xy(test_data, feature_columns)

        logger.info(f"OOS rows: {len(X_test)}  |  features: {len(feature_columns)}")

        if len(y_test) > 0:
            dist  = y_test.value_counts().sort_index()
            parts = "  ".join(
                f"label {int(lbl):+d}: {cnt} ({cnt/len(y_test)*100:.1f}%)"
                for lbl, cnt in dist.items()
            )
            logger.info(f"OOS labels — {parts}")

        logger.info("=" * 80)
        return X_test, y_test, feature_columns

    def get_summary(self) -> Dict[str, Any]:
        """Return serialisable summary of the walk-forward configuration and fold dates."""
        fold_dates = self._get_fold_dates()

        return {
            'window_type':        self.window_type,
            'train_start':        self.train_start.strftime('%Y-%m-%d'),
            'train_end':          self.train_end.strftime('%Y-%m-%d'),
            'test_start':         self.test_start.strftime('%Y-%m-%d'),
            'test_end':           self.test_end.strftime('%Y-%m-%d'),
            'n_splits':           len(fold_dates),
            'train_window_years': self.train_window_years,
            'val_window_years':   self.val_window_years,
            'folds': [
                {
                    'fold':        f['fold'],
                    'train_start': f['train_start'].strftime('%Y-%m-%d'),
                    'train_end':   f['train_end'].strftime('%Y-%m-%d'),
                    'val_start':   f['val_start'].strftime('%Y-%m-%d'),
                    'val_end':     f['val_end'].strftime('%Y-%m-%d'),
                }
                for f in fold_dates
            ],
        }
