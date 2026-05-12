#!/usr/bin/env python3
"""
Quarterly Rolling Walk-Forward Evaluation on OOS Period (2020-2024)

This script implements quarterly retraining with a rolling 3-year window:
- Q1 2020: Train 2017-01-01 to 2019-12-31, predict 2020-01-01 to 2020-03-31
- Q2 2020: Train 2017-04-01 to 2020-03-31, predict 2020-04-01 to 2020-06-30
- Q3 2020: Train 2017-07-01 to 2020-06-30, predict 2020-07-01 to 2020-09-30
- ... continuing through Q4 2024

This provides realistic evaluation by retraining quarterly on a rolling 3-year
window, allowing the model to adapt to recent market conditions while maintaining
consistent training data size.

Trains exclusively on SPY (S&P 500 ETF) using 10 stationary technical features.
Both long-short (primary) and long-only (secondary) strategies are evaluated.

Iteration 1: 10 pure technical features (baseline)
Iteration 2: 10 technical + 4 regime features (GMM regime detection)
"""

import argparse
import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.utils import resample

from interpret.glassbox import ExplainableBoostingClassifier

from src.ingestion.loader import DataLoader
from src.models.train import ModelTrainer
from src.models.regime_detection import RegimeDetector
from src.models.walk_forward import TECHNICAL_FEATURES

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# TECHNICAL_FEATURES imported from src.models.walk_forward — single source of truth.


def bootstrap_auc_ci(y_true, y_proba, n_boot=500, alpha=0.05, random_state=42):
    """
    Bootstrap 95% CI for ROC-AUC.  With ~60 obs per fold, the CI is wide
    (~±0.10-0.12) and essential for interpreting fold-level AUC values.

    Returns (lower, upper) at the (alpha/2, 1-alpha/2) percentiles.
    Returns (nan, nan) if y_true has only one class (AUC undefined).
    """
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    rng = np.random.RandomState(random_state)
    aucs = []
    for _ in range(n_boot):
        idx = resample(np.arange(len(y_true)), random_state=rng, stratify=y_true)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


def create_rolling_oos_folds():
    """
    Create quarterly rolling OOS folds for 2020-2024.
    
    Each fold uses a 3-year rolling training window and predicts one quarter.
    Training window slides forward by one quarter each fold.
    
    Returns list of dicts with train/test splits for each quarter.
    """
    folds = []
    fold_id = 1
    
    # Start dates for each quarter in 2020-2024
    oos_start = pd.Timestamp('2020-01-01')
    oos_end = pd.Timestamp('2024-12-31')
    
    # Generate quarterly folds
    current_test_start = oos_start
    
    while current_test_start < oos_end:
        # Test period: one quarter
        test_end = current_test_start + pd.DateOffset(months=3)
        
        # Training period: 3 years ending at test_start
        train_end = current_test_start
        train_start = train_end - pd.DateOffset(years=3)
        
        # Create quarter label (e.g., "2020-Q1")
        year = current_test_start.year
        quarter = (current_test_start.month - 1) // 3 + 1
        quarter_label = f"{year}-Q{quarter}"
        
        folds.append({
            'fold_id': fold_id,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': current_test_start,
            'test_end': test_end,
            'quarter_label': quarter_label
        })
        
        # Move to next quarter
        current_test_start = test_end
        fold_id += 1
    
    return folds


def get_feature_columns(data_df, iteration):
    """
    Get feature columns for the given iteration.
    
    Uses TECHNICAL_FEATURES list to ensure we use exactly the same 10 features
    as in walk_forward.py.
    
    Args:
        data_df: DataFrame to get columns from
        iteration: 1 for baseline, 2 for with regime features
        
    Returns:
        List of feature column names
    """
    # Start with technical features that exist in the data
    available_cols = data_df.columns.tolist()
    feature_cols = [col for col in TECHNICAL_FEATURES if col in available_cols]
    
    # For iteration 2, add regime features (dynamic: supports any n_components)
    if iteration == 2:
        regime_features = (
            ['regime_state'] +
            sorted([col for col in available_cols if col.startswith('regime_prob_')])
        )
        for col in regime_features:
            if col in available_cols:
                feature_cols.append(col)
    
    return feature_cols


def load_is_last_fold_models(iteration, best_T, suffix=''):
    """
    Load last-fold models from IS walk-forward results.

    The IS walk-forward (2008-2020) produces the initial models. These are
    applied directly to the first OOS quarter (Q1 2020) without retraining,
    following the tutor's pipeline design: "se llega al backtest con los
    modelos entrenados, no se empieza de cero".

    Args:
        iteration: 1 for baseline, 2 for regime features
        best_T: distillation temperature (used to select EBM distilled)

    Returns:
        Dict with IS models, or None if PKL not found
    """
    if iteration == 1:
        path = Path(f'data/processed/walk_forward_results{suffix}.pkl')
        distill_path = Path(f'data/processed/walk_forward_distillation_results{suffix}.pkl')

        if not path.exists():
            logger.warning(f"IS results not found: {path} — first OOS fold will retrain from scratch")
            return None

        with open(path, 'rb') as f:
            wf = pickle.load(f)

        last = wf['all_fold_results'][-1]
        is_models = {
            'lightgbm': last['models']['lightgbm'],
            'ebm_primary': last['models'].get('ebm_primary'),
            'ebm_distilled': None,
            'feature_names': last['feature_names'],
            'fold_number': last['fold_number'],
        }
        if is_models['ebm_primary'] is not None:
            logger.info(f"IS EBM primary loaded from fold {last['fold_number']}")

        if distill_path.exists():
            with open(distill_path, 'rb') as f:
                distill = pickle.load(f)
            last_d = distill['all_fold_results'][-1]
            is_models['ebm_distilled'] = last_d['ebm_distilled_models'].get(best_T)
            if is_models['ebm_distilled'] is not None:
                logger.info(f"IS EBM distilled (T={best_T}) loaded from distillation fold {last_d['fold_number']}")
        else:
            logger.warning("Distillation PKL not found — EBM distilled will be retrained for first OOS fold")

        logger.info(f"IS Iter 1 models loaded from fold {last['fold_number']} "
                    f"(features: {is_models['feature_names']})")
        return is_models

    elif iteration == 2:
        path = Path(f'data/processed/walk_forward_results_regime{suffix}.pkl')

        if not path.exists():
            logger.warning(f"IS regime results not found: {path} — first OOS fold will retrain from scratch")
            return None

        with open(path, 'rb') as f:
            wf = pickle.load(f)

        last = wf['all_fold_results'][-1]
        is_models = {
            'lightgbm': last['models']['lightgbm'],
            'ebm_primary': last['models'].get('ebm_primary'),
            'ebm_distilled': None,  # trained using IS LightGBM soft labels in first fold
            # regime_detector is intentionally NOT loaded from PKL — the serialized
            # detector may have a stale n_regimes that differs from config.yaml.
            # Q1 2020 will refit a fresh RegimeDetector from config on its training window.
            'feature_names': last['feature_names'],
            'fold_number': last['fold_number'],
        }
        if is_models['ebm_primary'] is not None:
            logger.info(f"IS EBM primary loaded from fold {last['fold_number']}")

        logger.info(f"IS Iter 2 models loaded from fold {last['fold_number']} "
                    f"(features: {is_models['feature_names']})")
        return is_models

    return None


def train_fold(data, fold_info, config, iteration, best_T=4, is_models=None, best_threshold=0.50):
    """Train models for a single fold."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FOLD {fold_info['fold_id']}: {fold_info['quarter_label']}")
    logger.info(f"Train: {fold_info['train_start'].date()} to {fold_info['train_end'].date()}")
    logger.info(f"Test:  {fold_info['test_start'].date()} to {fold_info['test_end'].date()}")
    logger.info(f"{'=' * 80}")
    
    # Filter data for this fold
    train_data = data[
        (data.index.get_level_values('date') >= fold_info['train_start']) &
        (data.index.get_level_values('date') < fold_info['train_end'])
    ].copy()

    # Purge last max_holding_period trading days (López de Prado, AFML ch.7).
    # Labels near the train/test boundary use prices that fall inside the test period.
    purge_days = config.get('features', {}).get('triple_barrier', {}).get('max_holding_period', 8)
    if purge_days > 0:
        train_dates = sorted(train_data.index.get_level_values('date').unique())
        if len(train_dates) > purge_days:
            cutoff = train_dates[-purge_days]
            train_data = train_data[train_data.index.get_level_values('date') < cutoff]

    test_data = data[
        (data.index.get_level_values('date') >= fold_info['test_start']) &
        (data.index.get_level_values('date') < fold_info['test_end'])
    ].copy()
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Handle regime features for iteration 2
    if iteration == 2:
        if is_models is not None and is_models.get('regime_detector') is not None:
            # This branch is no longer reachable: load_is_last_fold_models() no longer
            # stores the PKL detector (stale n_regimes risk). Kept as a safety fallback.
            logger.info("First OOS fold: using IS GMM regime detector (no refitting)...")
            regime_detector = is_models['regime_detector']
            logger.info(f"RegimeDetector: n_components={regime_detector.n_regimes}")
        else:
            logger.info("Fitting GMM regime detector on training data...")
            regime_detector = RegimeDetector(config)
            logger.info(f"RegimeDetector: n_components={regime_detector.n_regimes}")
            # Fit GMM on training data
            regime_detector.fit(train_data)
        
        # Get regime labels and probabilities for training data
        train_regime_labels, train_regime_probs = regime_detector.predict(train_data)
        
        # Add regime features to training data (loop covers any n_components)
        train_data['regime_state'] = train_regime_labels
        for i in range(train_regime_probs.shape[1]):
            train_data[f'regime_prob_{i}'] = train_regime_probs[:, i]

        # Predict regimes for test data
        test_regime_labels, test_regime_probs = regime_detector.predict(test_data)

        test_data['regime_state'] = test_regime_labels
        for i in range(test_regime_probs.shape[1]):
            test_data[f'regime_prob_{i}'] = test_regime_probs[:, i]
    
    # Get feature columns
    # For first OOS fold using IS models: use the exact feature set the IS model was trained on.
    # This handles IC-selected subsets: IS model may have been trained on fewer than 13 features.
    if is_models is not None:
        feature_cols = is_models['feature_names']
        logger.info(f"First OOS fold: using IS feature set ({len(feature_cols)} features)")
        # NOTE: The IS model may have fewer features than subsequent OOS folds if IC
        # selection was applied during the IS walk-forward (e.g. macd_signal removed
        # when IC < 0.01).  From Q2 2020 onwards, get_feature_columns returns the
        # full fixed set (10 or 14 features) without IC filtering.  This one-fold
        # feature-set discontinuity is intentional: the IS model is used as-is, and
        # the new model retrained in Q2 2020 starts fresh with the full feature set.
    else:
        feature_cols = get_feature_columns(train_data, iteration)

    logger.info(f"Features selected: {len(feature_cols)}")
    logger.info(f"Feature list: {feature_cols}")

    # ASSERTION: Verify feature count (skip for IS models — may use IC-selected subset)
    _n_regime = config.get('models', {}).get('regime', {}).get('n_components', 3)
    # 10 technical + 1 regime_state + n_components regime_prob columns
    expected_count = 10 if iteration == 1 else (10 + 1 + _n_regime)
    if is_models is None and len(feature_cols) != expected_count:
        error_msg = f"Feature count mismatch! Expected {expected_count}, got {len(feature_cols)}"
        logger.error(error_msg)
        logger.error(f"Features: {feature_cols}")

        # Check which columns are extra or missing
        if iteration == 1:
            expected_features = TECHNICAL_FEATURES
        else:
            expected_features = (
                TECHNICAL_FEATURES +
                ['regime_state'] +
                [f'regime_prob_{i}' for i in range(_n_regime)]
            )

        extra = set(feature_cols) - set(expected_features)
        missing = set(expected_features) - set(feature_cols)

        if extra:
            logger.error(f"Extra columns: {extra}")
        if missing:
            logger.error(f"Missing columns: {missing}")

        raise ValueError(error_msg)
    
    logger.success(f"✓ Feature count verified: {len(feature_cols)} features")
    
    # Drop rows with NaN labels or NaN features before training.
    # NaN labels appear during the warm-up period (vol_20d needs ~20 rows)
    # and at the end of the series (last 8 rows have NaN label_binary, max_holding_period).
    # label_binary: triple barrier — +1 = take profit, -1 = stop loss or time barrier.
    valid_train = train_data['label_binary'].notna() & train_data[feature_cols].notna().all(axis=1)
    train_clean = train_data[valid_train]

    valid_test = test_data['label_binary'].notna() & test_data[feature_cols].notna().all(axis=1)
    test_clean = test_data[valid_test]

    logger.info(f"Train samples after NaN drop: {len(train_clean)} (dropped {len(train_data) - len(train_clean)})")
    logger.info(f"Test samples after NaN drop:  {len(test_clean)} (dropped {len(test_data) - len(test_clean)})")

    # Prepare arrays for training
    X_train = train_clean[feature_cols].values
    y_train_binary = (train_clean['label_binary'] == 1).astype(int).values

    X_test = test_clean[feature_cols].values
    y_test_binary = (test_clean['label_binary'] == 1).astype(int).values

    logger.info(f"Train label distribution: {np.bincount(y_train_binary)}")
    logger.info(f"Test label distribution: {np.bincount(y_test_binary)}")
    
    trainer = ModelTrainer(config)

    if is_models is not None:
        # First OOS fold: carry IS walk-forward models forward — no retraining.
        logger.info(f"First OOS fold: using IS models from fold {is_models['fold_number']} "
                    f"(LightGBM + EBM primary) — no retraining")
        lgbm_model = is_models['lightgbm']
        ebm_primary_model = is_models.get('ebm_primary')
    else:
        # Subsequent OOS folds: retrain on the 3-year rolling window
        logger.info("Training LightGBM...")
        lgbm_model = trainer._train_lightgbm(X_train, y_train_binary, fold_info['fold_id'])
        logger.info("Training EBM primary (hard labels)...")
        ebm_primary_model = trainer._train_ebm(X_train, y_train_binary, fold_info['fold_id'])

    # EBM distilled (knowledge distillation from LightGBM)
    if is_models is not None and is_models.get('ebm_distilled') is not None:
        # First OOS fold Iter 1: IS distilled EBM available
        logger.info(f"First OOS fold: using IS distilled EBM (T={best_T})")
        ebm_dist_model = is_models['ebm_distilled']
    else:
        # All other cases: train EBM distilled using lgbm_model soft labels.
        # For first OOS fold Iter 2 (no IS distilled): lgbm_model is the IS LightGBM,
        # so distillation still uses IS knowledge — consistent with tutor's design.
        logger.info(f"Training EBM distilled (T={best_T}, threshold={best_threshold:.2f})...")
        teacher_proba = lgbm_model.predict_proba(X_train)[:, 1]

        # Temperature scaling — best_T selected on IS period, no look-ahead
        p = np.clip(teacher_proba, 1e-8, 1 - 1e-8)
        logits = np.log(p / (1 - p))
        soft_pos = 1 / (1 + np.exp(-logits / best_T))

        # Confidence threshold filtering: discard ambiguous training observations.
        # Filter is applied to raw teacher_proba (before temperature scaling) so
        # that the confidence criterion is independent of the temperature parameter.
        if best_threshold > 0.50:
            thr_mask = (teacher_proba > best_threshold) | (teacher_proba < (1.0 - best_threshold))
            n_kept = int(thr_mask.sum())
            n_total = len(teacher_proba)
            logger.info(
                f"  Confidence filter: {n_kept}/{n_total} kept "
                f"({n_total - n_kept} discarded, "
                f"{100 * (n_total - n_kept) / n_total:.1f}%)"
            )
            X_fit = X_train[thr_mask]
            soft_pos_fit = soft_pos[thr_mask]
        else:
            X_fit = X_train
            soft_pos_fit = soft_pos

        y_hard = (soft_pos_fit >= 0.5).astype(int)
        sample_weight = np.abs(soft_pos_fit - 0.5) * 2

        ebm_config = config.get('models', {}).get('ebm', {})
        ebm_dist_model = ExplainableBoostingClassifier(
            max_bins=ebm_config.get('max_bins', 128),
            max_interaction_bins=ebm_config.get('max_interaction_bins', 32),
            interactions=ebm_config.get('interactions', 10),
            learning_rate=ebm_config.get('learning_rate', 0.01),
            max_rounds=ebm_config.get('max_rounds', 5000),
            min_samples_leaf=ebm_config.get('min_samples_leaf', 10),
            random_state=ebm_config.get('random_state', 42)
        )
        ebm_dist_model.fit(X_fit, y_hard, sample_weight=sample_weight)
        logger.success("EBM distilled trained")
    
    # Evaluate on test set (SPY only)
    results = {}

    ticker = 'SPY'
    ticker_mask = test_clean.index.get_level_values('ticker') == ticker
    X_test_ticker = X_test[ticker_mask]
    y_test_ticker = y_test_binary[ticker_mask]
    
    # Get predictions
    lgbm_pred_proba = lgbm_model.predict_proba(X_test_ticker)[:, 1]
    lgbm_pred = (lgbm_pred_proba >= 0.5).astype(int)

    ebm_dist_pred_proba = ebm_dist_model.predict_proba(X_test_ticker)[:, 1]
    ebm_dist_pred = (ebm_dist_pred_proba >= 0.5).astype(int)

    ebm_primary_pred_proba = ebm_primary_model.predict_proba(X_test_ticker)[:, 1] if ebm_primary_model is not None else np.full(len(X_test_ticker), 0.5)
    ebm_primary_pred = (ebm_primary_pred_proba >= 0.5).astype(int)

    lgbm_auc_lo,        lgbm_auc_hi        = bootstrap_auc_ci(y_test_ticker, lgbm_pred_proba)
    ebm_auc_lo,         ebm_auc_hi         = bootstrap_auc_ci(y_test_ticker, ebm_dist_pred_proba)
    ebm_primary_auc_lo, ebm_primary_auc_hi = bootstrap_auc_ci(y_test_ticker, ebm_primary_pred_proba)

    results[ticker] = {
        'lightgbm': {
            'accuracy': accuracy_score(y_test_ticker, lgbm_pred),
            'roc_auc': roc_auc_score(y_test_ticker, lgbm_pred_proba),
            'roc_auc_ci': (lgbm_auc_lo, lgbm_auc_hi),
            'f1': f1_score(y_test_ticker, lgbm_pred, zero_division=0),
            'brier_score': brier_score_loss(y_test_ticker, lgbm_pred_proba),
            'predictions': lgbm_pred,
            'pred_proba': lgbm_pred_proba
        },
        'ebm_primary': {
            'accuracy': accuracy_score(y_test_ticker, ebm_primary_pred),
            'roc_auc': roc_auc_score(y_test_ticker, ebm_primary_pred_proba),
            'roc_auc_ci': (ebm_primary_auc_lo, ebm_primary_auc_hi),
            'f1': f1_score(y_test_ticker, ebm_primary_pred, zero_division=0),
            'brier_score': brier_score_loss(y_test_ticker, ebm_primary_pred_proba),
            'predictions': ebm_primary_pred,
            'pred_proba': ebm_primary_pred_proba
        },
        'ebm_distilled': {
            'accuracy': accuracy_score(y_test_ticker, ebm_dist_pred),
            'roc_auc': roc_auc_score(y_test_ticker, ebm_dist_pred_proba),
            'roc_auc_ci': (ebm_auc_lo, ebm_auc_hi),
            'f1': f1_score(y_test_ticker, ebm_dist_pred, zero_division=0),
            'brier_score': brier_score_loss(y_test_ticker, ebm_dist_pred_proba),
            'predictions': ebm_dist_pred,
            'pred_proba': ebm_dist_pred_proba
        },
        'y_true': y_test_ticker,
        'test_data': test_clean[ticker_mask]
    }

    logger.info(
        f"{ticker} - LightGBM:    Acc={results[ticker]['lightgbm']['accuracy']:.3f}, "
        f"AUC={results[ticker]['lightgbm']['roc_auc']:.3f} "
        f"[{lgbm_auc_lo:.3f}, {lgbm_auc_hi:.3f}]"
    )
    logger.info(
        f"{ticker} - EBM Primary: Acc={results[ticker]['ebm_primary']['accuracy']:.3f}, "
        f"AUC={results[ticker]['ebm_primary']['roc_auc']:.3f} "
        f"[{ebm_primary_auc_lo:.3f}, {ebm_primary_auc_hi:.3f}]"
    )
    logger.info(
        f"{ticker} - EBM Distilled: Acc={results[ticker]['ebm_distilled']['accuracy']:.3f}, "
        f"AUC={results[ticker]['ebm_distilled']['roc_auc']:.3f} "
        f"[{ebm_auc_lo:.3f}, {ebm_auc_hi:.3f}]"
    )
    
    return results


def calculate_trading_metrics(predictions, test_data, strategy='long_only'):
    """Calculate trading metrics from predictions."""
    # Get returns (use ret_1d_forward to avoid look-ahead bias)
    returns = test_data['ret_1d_forward'].values
    
    # Drop NaN values (last row per ticker)
    nan_mask = ~np.isnan(returns)
    returns = returns[nan_mask]
    predictions = predictions[nan_mask]
    
    # Convert predictions to signals
    if strategy == 'long_only':
        # Long when predict 1, cash when predict 0
        signals = predictions.copy()
    else:  # long_short
        # Long when predict 1, short when predict 0
        signals = predictions.copy()
        signals[signals == 0] = -1
    
    # Calculate strategy returns
    strategy_returns = signals * returns
    
    # Calculate cumulative return
    cumulative_return = (1 + strategy_returns).prod() - 1
    
    # Calculate Sharpe ratio (annualized, assuming daily returns)
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': cumulative_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }


def main():
    """Run quarterly rolling OOS evaluation."""
    parser = argparse.ArgumentParser(description='Quarterly rolling OOS evaluation')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to config YAML (default: config/config.yaml)')
    parser.add_argument(
        '--mode', default='standard', choices=['standard', 'thr'],
        help=(
            'standard (default): temperature-scaled EBM distillation; '
            'thr: apply confidence threshold filter when training EBM '
            '(loads best_threshold from walk_forward_distillation_results_thr.pkl)'
        ),
    )
    args = parser.parse_args()

    config_stem = Path(args.config).stem
    suffix = config_stem[len('config'):]
    mode_suffix = '_thr' if args.mode == 'thr' else ''

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"logs/rolling_oos_quarterly_{timestamp}.log"

    logger.add(
        log_file,
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("=" * 80)
    logger.info("QUARTERLY ROLLING WALK-FORWARD EVALUATION: OOS PERIOD 2020-2024")
    logger.info("=" * 80)
    logger.info("Q1 2020: IS walk-forward models carried forward (no retraining)")
    logger.info("Q2 2020 onwards: quarterly retraining with 3-year rolling window")
    logger.info("Training exclusively on SPY with 10 stationary technical features")
    logger.info(f"Config: {args.config}  (output suffix: '{suffix}')")
    logger.info(
        "NOTE: the config suffix only affects the IS seed model used for Q1 2020. "
        "From Q2 2020 onward, both configs retrain on the same 3-year rolling window "
        "with random_state=42, so OOS metrics from Q2 2020+ are identical across configs. "
        "Meaningful config differences exist only in the IS walk-forward results."
    )
    logger.info("=" * 80)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load best temperature (and threshold if mode='thr') from distillation results
    if args.mode == 'thr':
        distill_path = Path(f'data/processed/walk_forward_distillation_results{suffix}_thr.pkl')
    else:
        distill_path = Path(f'data/processed/walk_forward_distillation_results{suffix}.pkl')

    best_threshold = 0.50  # default: no filtering
    if distill_path.exists():
        with open(distill_path, 'rb') as f:
            distill_results = pickle.load(f)
        best_T = distill_results['best_T']
        if args.mode == 'thr':
            best_threshold = distill_results.get('best_threshold', 0.55)
            logger.info(
                f"Loaded from {distill_path.name}: T={best_T}, threshold={best_threshold:.2f}"
            )
        else:
            logger.info(f"Best temperature loaded from distillation results: T={best_T}")
    else:
        best_T = 1 if args.mode == 'thr' else 4
        logger.warning(
            f"Distillation results not found ({distill_path}) — "
            f"using defaults T={best_T}, threshold={best_threshold}"
        )

    # Load data
    logger.info("\nLoading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()
    
    logger.info(f"Data loaded: {data.shape}")
    logger.info(f"Date range: {data.index.get_level_values('date').min()} to {data.index.get_level_values('date').max()}")
    
    # Create folds
    folds = create_rolling_oos_folds()
    logger.info(f"\nGenerated {len(folds)} quarterly folds")
    
    # Run both iterations
    all_results = {
        'iteration_1': {},  # Baseline
        'iteration_2': {}   # With regime features
    }
    
    # Load IS walk-forward models for the first OOS fold.
    # The IS walk-forward (2008-2020) produces the initial models; Q1 2020 uses
    # them directly. From Q2 2020 onwards, models are retrained quarterly.
    # For mode='thr': IS EBM is forced to None so Q1 2020 retrains with threshold
    # filtering applied — keeps distillation consistent across all OOS folds.
    logger.info("\nLoading IS walk-forward models for first OOS fold (Q1 2020)...")
    is_models_per_iter = {
        1: load_is_last_fold_models(1, best_T, suffix),
        2: load_is_last_fold_models(2, best_T, suffix),
    }
    if args.mode == 'thr':
        for it in [1, 2]:
            if is_models_per_iter[it] is not None:
                is_models_per_iter[it]['ebm_distilled'] = None
        logger.info("thr mode: IS EBM set to None — Q1 2020 will retrain with threshold filter")

    for iteration in [1, 2]:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ITERATION {iteration}: {'BASELINE (10 stationary technical features)' if iteration == 1 else 'WITH REGIME FEATURES (10 technical + 4 regime)'}")
        logger.info(f"{'=' * 80}")

        is_models_iter = is_models_per_iter[iteration]
        iteration_results = []

        for fold_idx, fold_info in enumerate(folds):
            # Pass IS models only for the first fold (Q1 2020)
            is_models = is_models_iter if fold_idx == 0 else None
            fold_results = train_fold(
                data, fold_info, config, iteration,
                best_T=best_T, is_models=is_models, best_threshold=best_threshold,
            )
            fold_results['fold_info'] = fold_info
            iteration_results.append(fold_results)
        
        # Aggregate results across all folds (SPY only)
        aggregated = {
            'SPY': {'lightgbm': {}, 'ebm_primary': {}, 'ebm_distilled': {}}
        }

        ticker = 'SPY'
        for model in ['lightgbm', 'ebm_primary', 'ebm_distilled']:
            # Collect all predictions and true labels
            all_preds = []
            all_proba = []
            all_true = []
            all_test_data = []
            
            for fold_results in iteration_results:
                all_preds.append(fold_results[ticker][model]['predictions'])
                all_proba.append(fold_results[ticker][model]['pred_proba'])
                all_true.append(fold_results[ticker]['y_true'])
                all_test_data.append(fold_results[ticker]['test_data'])
            
            # Concatenate
            all_preds = np.concatenate(all_preds)
            all_proba = np.concatenate(all_proba)
            all_true = np.concatenate(all_true)
            all_test_data = pd.concat(all_test_data)
            
            # Calculate overall metrics
            aggregated[ticker][model]['accuracy'] = accuracy_score(all_true, all_preds)
            aggregated[ticker][model]['roc_auc'] = roc_auc_score(all_true, all_proba)
            aggregated[ticker][model]['f1'] = f1_score(all_true, all_preds, zero_division=0)
            aggregated[ticker][model]['brier_score'] = brier_score_loss(all_true, all_proba)

            # Calculate trading metrics for both strategies
            # Long-short is the primary strategy (acts on both signals)
            # Long-only is shown as a robustness check (cash when signal=-1)
            aggregated[ticker][model]['trading_longshort'] = calculate_trading_metrics(
                all_preds, all_test_data, strategy='long_short'
            )
            aggregated[ticker][model]['trading_longonly'] = calculate_trading_metrics(
                all_preds, all_test_data, strategy='long_only'
            )
        
        all_results[f'iteration_{iteration}'] = {
            'fold_results': iteration_results,
            'aggregated': aggregated
        }
    
    # Save results
    output_dir = Path('data/processed/rolling_oos')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'rolling_oos_quarterly_results{suffix}{mode_suffix}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    logger.success(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("QUARTERLY ROLLING OOS RESULTS SUMMARY (2020-2024)")
    print("=" * 80)
    print(f"Total quarters evaluated: {len(folds)}")
    print(f"Training window: 3 years (rolling)")
    print(f"Retraining frequency: Quarterly")
    print("=" * 80)
    
    for iteration in [1, 2]:
        print(f"\n{'ITERATION ' + str(iteration)}: {'BASELINE (10 features)' if iteration == 1 else 'WITH REGIME (10 + 4 features)'}")
        print("-" * 80)

        agg = all_results[f'iteration_{iteration}']['aggregated']

        ticker = 'SPY'
        print(f"\n{ticker}:")
        for model in ['lightgbm', 'ebm_primary', 'ebm_distilled']:
            metrics = agg[ticker][model]
            model_name = {'lightgbm': 'LIGHTGBM', 'ebm_primary': 'EBM PRIMARY', 'ebm_distilled': 'EBM DISTILLED'}[model]
            ls = metrics['trading_longshort']
            lo = metrics['trading_longonly']
            print(f"  {model_name}:")
            print(f"    Accuracy:           {metrics['accuracy']:.3f}")
            print(f"    ROC-AUC:            {metrics['roc_auc']:.3f}")
            print(f"    F1 Score:           {metrics['f1']:.3f}")
            print(f"    Brier Score:        {metrics['brier_score']:.4f}")
            print(f"    [Long-Short] Return:{ls['total_return']:>8.2%}  Sharpe:{ls['sharpe']:>5.2f}  MaxDD:{ls['max_drawdown']:>7.2%}")
            print(f"    [Long-Only]  Return:{lo['total_return']:>8.2%}  Sharpe:{lo['sharpe']:>5.2f}  MaxDD:{lo['max_drawdown']:>7.2%}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("QUARTERLY ROLLING OOS EVALUATION COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
