#!/usr/bin/env python3
"""
RuleFit Regime — Interpretable Rules from Regime-Aware LightGBM (2-stage distillation)

Implements the full Iter2 distillation chain:
    LightGBM Regime (teacher, 15 features)
        → EBM Distilled Regime (student, 15 features, trained inline)
        → RuleFit (trained on EBM Distilled soft labels → explicit if-then rules)

This mirrors the Iter1 chain (LightGBM → EBM Distilled → RuleFit) but with
15 features (11 technical + 4 regime) instead of 11 technical features.

Features used (15):
    11 technical: ret_5d, ret_21d, vol_20d, rsi_14, macd_line, macd_signal,
                  macd_hist, bb_pct, bb_width, atr_14, volume_ratio
    4 regime (GMM, human-readable labels):
        regime        → regime_state  (0=Bull, 1=Neutral, 2=Bear)
        prob_bull     → regime_prob_0 (probability of Bull regime)
        prob_neutral  → regime_prob_1 (probability of Neutral regime)
        prob_bear     → regime_prob_2 (probability of Bear regime)

Training data: IS last 2 folds (2015-2019).
Evaluation: full OOS period 2020-2024.

Usage:
    python run_rulefit_regime.py --config config/config_2010.yaml
"""

import argparse
import warnings
import re
import yaml
import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss
from interpret.glassbox import ExplainableBoostingClassifier

from src.ingestion.loader import DataLoader
from src.models.train import ModelTrainer

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Human-readable names for regime features
REGIME_RENAME = {
    'regime_state':  'regime',        # 0=Bull, 1=Neutral, 2=Bear
    'regime_prob_0': 'prob_bull',
    'regime_prob_1': 'prob_neutral',
    'regime_prob_2': 'prob_bear',
}
REGIME_LEGEND = "regime: 0=Bull (low vol, positive returns)  1=Neutral  2=Bear (high vol, VIX spike)"


def calculate_trading_metrics(predictions: np.ndarray, returns: np.ndarray, strategy: str) -> dict:
    """Calculate trading metrics for long-short or long-only strategy."""
    nan_mask = ~np.isnan(returns)
    returns = returns[nan_mask]
    predictions = predictions[nan_mask]

    if strategy == 'long_only':
        signals = predictions.copy()
    else:
        signals = np.where(predictions == 1, 1, -1)

    strategy_returns = signals * returns
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0.0
    cumulative = (1 + strategy_returns).cumprod()
    max_drawdown = ((cumulative - np.maximum.accumulate(cumulative)) / np.maximum.accumulate(cumulative)).min()

    return {'total_return': total_return, 'sharpe': sharpe, 'max_drawdown': max_drawdown}


def main():
    parser = argparse.ArgumentParser(description='RuleFit from regime-aware LightGBM soft labels')
    parser.add_argument('--config', default='config/config_2010.yaml',
                        help='Path to config YAML (default: config/config_2010.yaml)')
    args = parser.parse_args()

    config_stem = Path(args.config).stem
    suffix = config_stem[len('config'):]

    logger.add(
        "logs/rulefit_regime_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("=" * 80)
    logger.info("RULEFIT REGIME: INTERPRETABLE RULES FROM REGIME-AWARE LIGHTGBM")
    logger.info("=" * 80)
    logger.info("Chain: LightGBM Regime (15 features) → EBM Distilled Regime → RuleFit")
    logger.info(f"Config: {args.config}  (suffix: '{suffix}')")
    logger.info(REGIME_LEGEND)
    logger.info("=" * 80)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Step 1: Load regime walk-forward PKL — last IS fold LightGBM
    # ------------------------------------------------------------------
    regime_path = Path(f'data/processed/walk_forward_results_regime{suffix}.pkl')
    if not regime_path.exists():
        raise FileNotFoundError(
            f"Regime results not found: {regime_path}\n"
            "Run run_walk_forward_regime.py first."
        )

    with open(regime_path, 'rb') as f:
        regime_data = pickle.load(f)

    last_fold = regime_data['all_fold_results'][-1]
    lgbm_model = last_fold['models']['lightgbm']
    regime_detector = last_fold['regime_detector']
    train_start = last_fold['train_start']
    train_end = last_fold['train_end']
    raw_feature_names = last_fold['feature_names']  # original names with regime_state etc.
    fold_number = last_fold['fold_number']

    # Apply human-readable renaming
    feature_names = [REGIME_RENAME.get(f, f) for f in raw_feature_names]

    logger.info(f"IS last fold: {fold_number}  ({train_start.date()} → {train_end.date()})")
    logger.info(f"Raw features ({len(raw_feature_names)}): {raw_feature_names}")
    logger.info(f"Display features ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # Step 2: Load data and add regime features
    # ------------------------------------------------------------------
    logger.info("\nLoading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()

    if 'ticker' in data.index.names:
        spy = data.xs('SPY', level='ticker')
    else:
        spy = data

    # Add regime features using the detector from the last fold
    logger.info("Adding regime features...")
    regime_state, regime_proba = regime_detector.predict(spy)
    spy_with_regime = regime_detector.add_regime_features(spy, regime_state, regime_proba)

    # ------------------------------------------------------------------
    # Step 3: Recreate IS training split (extended window: 2015-2019)
    # ------------------------------------------------------------------
    extended_train_start = train_start - pd.DateOffset(years=1)
    train_data = spy_with_regime[
        (spy_with_regime.index >= extended_train_start) &
        (spy_with_regime.index < train_end)
    ].copy()
    logger.info(f"Extended IS window: {extended_train_start.date()} -> {train_end.date()} (original: {train_start.date()})")

    valid = (
        train_data['label_binary'].notna() &
        train_data[raw_feature_names].notna().all(axis=1)
    )
    train_clean = train_data[valid]
    X_train = train_clean[raw_feature_names].values

    logger.info(f"Training samples: {len(X_train)}  (limit: 800 — {'will subsample' if len(X_train) > 800 else 'OK'})")

    # ------------------------------------------------------------------
    # Step 4a: Train EBM Distilled Regime (LightGBM → EBM, T=1)
    # ------------------------------------------------------------------
    logger.info("\nStep 4a: Training EBM Distilled Regime (LightGBM Regime → EBM Distilled)...")

    lgbm_proba = lgbm_model.predict_proba(X_train)[:, 1]
    p = np.clip(lgbm_proba, 1e-8, 1 - 1e-8)
    # T=1: no temperature scaling — same as run_rolling_oos_evaluation.py with best_T=1
    soft_lgbm = p

    y_hard_lgbm = (soft_lgbm >= 0.5).astype(int)
    sw_lgbm = np.abs(soft_lgbm - 0.5) * 2

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
    ebm_dist_model.fit(X_train, y_hard_lgbm, sample_weight=sw_lgbm)
    logger.success("EBM Distilled Regime trained.")

    # ------------------------------------------------------------------
    # Step 4b: Generate soft labels from EBM Distilled Regime
    # ------------------------------------------------------------------
    logger.info("\nStep 4b: Generating soft labels from EBM Distilled Regime...")

    ebm_proba = ebm_dist_model.predict_proba(X_train)[:, 1]
    p = np.clip(ebm_proba, 1e-8, 1 - 1e-8)
    soft_pos = p  # T=1

    y_hard = (soft_pos >= 0.5).astype(int)
    sample_weight = np.abs(soft_pos - 0.5) * 2

    n_pos = y_hard.sum()
    n_neg = len(y_hard) - n_pos
    logger.info(f"EBM soft labels: min={soft_pos.min():.3f}  max={soft_pos.max():.3f}  mean={soft_pos.mean():.3f}")
    logger.info(f"Hard labels: {n_pos} positive ({n_pos/len(y_hard)*100:.1f}%), {n_neg} negative")

    # ------------------------------------------------------------------
    # Step 5: Train RuleFit with readable feature names
    # ------------------------------------------------------------------
    logger.info("\nTraining RuleFit on EBM Distilled Regime soft labels...")

    trainer = ModelTrainer(config)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)  # use readable names

    result = trainer._train_rulefit(
        X_train_df, y_hard, fold_number, feature_names, sample_weight=sample_weight
    )
    if result is None:
        raise RuntimeError("RuleFit training failed — check logs for details.")

    rulefit_model, feature_mapping = result
    logger.success("RuleFit trained successfully.")

    # ------------------------------------------------------------------
    # Step 6: Extract and display readable rules
    # ------------------------------------------------------------------
    logger.info("\nExtracting rules...")

    rules_df = rulefit_model._get_rules().copy()

    def translate_rule(rule_str, mapping):
        """Replace integer feature indices with original names — single-pass to avoid cascade."""
        sorted_items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
        pattern = '|'.join(re.escape(idx) for idx, _ in sorted_items)
        full_pattern = rf'(?<![.\d])({pattern})(?![.\d])'
        return re.sub(full_pattern, lambda m: mapping[m.group(0)], rule_str)

    rules_df['rule'] = rules_df['rule'].apply(lambda r: translate_rule(str(r), feature_mapping))

    active_rules = rules_df[
        (rules_df['coef'].abs() > 0) &
        (rules_df['support'] > 0.01)
    ].copy()
    active_rules = active_rules.sort_values('coef', key=abs, ascending=False)

    is_rule = active_rules['type'] == 'rule'
    rule_part = active_rules[is_rule].head(20)
    linear_part = active_rules[~is_rule].head(15)

    print("\n" + "=" * 80)
    print("TOP IF-THEN RULES (sorted by |coefficient|)")
    print("=" * 80)
    print(f"  {REGIME_LEGEND}")
    print(f"  {'#':<4} {'Coef':>8}  {'Support':>8}  {'Dir':<7}  Rule")
    print("  " + "-" * 72)
    for i, (_, row) in enumerate(rule_part.iterrows(), 1):
        direction = "LONG" if row['coef'] > 0 else "SHORT"
        print(f"  {i:<4} {row['coef']:>+8.4f}  {row['support']:>7.1%}  {direction:<7}  {row['rule']}")

    print(f"\n{'='*80}")
    print("LINEAR TERMS (global feature effects)")
    print("=" * 80)
    print(f"  {'Feature':<20} {'Coef':>8}  {'Support':>8}  Direction")
    print("  " + "-" * 52)
    for _, row in linear_part.iterrows():
        direction = "bullish" if row['coef'] > 0 else "bearish"
        print(f"  {row['rule']:<20} {row['coef']:>+8.4f}  {row['support']:>7.1%}  {direction}")

    # ------------------------------------------------------------------
    # Step 7: Evaluate on OOS period (2020-2024)
    # ------------------------------------------------------------------
    logger.info("\nEvaluating RuleFit on OOS period 2020-2024...")

    oos_data = spy_with_regime[
        (spy_with_regime.index >= '2020-01-01') &
        (spy_with_regime.index <= '2024-12-31')
    ].copy()

    oos_valid = (
        oos_data['label_binary'].notna() &
        oos_data['ret_1d_forward'].notna() &
        oos_data[raw_feature_names].notna().all(axis=1)
    )
    oos_clean = oos_data[oos_valid]
    X_oos = oos_clean[raw_feature_names].values
    y_oos = (oos_clean['label_binary'] == 1).astype(int).values
    returns_oos = oos_clean['ret_1d_forward'].values

    oos_pred_proba = rulefit_model.predict_proba(X_oos)[:, 1]
    oos_pred = (oos_pred_proba >= 0.5).astype(int)

    oos_metrics = {
        'accuracy': accuracy_score(y_oos, oos_pred),
        'roc_auc': roc_auc_score(y_oos, oos_pred_proba),
        'f1': f1_score(y_oos, oos_pred, zero_division=0),
        'brier_score': brier_score_loss(y_oos, oos_pred_proba),
        'trading_longshort': calculate_trading_metrics(oos_pred, returns_oos, 'long_short'),
        'trading_longonly': calculate_trading_metrics(oos_pred, returns_oos, 'long_only'),
    }

    print(f"\n{'='*80}")
    print("OOS EVALUATION (2020-2024)")
    print("=" * 80)
    print(f"  Accuracy:     {oos_metrics['accuracy']:.3f}")
    print(f"  ROC-AUC:      {oos_metrics['roc_auc']:.3f}")
    print(f"  F1 Score:     {oos_metrics['f1']:.3f}")
    print(f"  Brier Score:  {oos_metrics['brier_score']:.4f}")
    ls = oos_metrics['trading_longshort']
    lo = oos_metrics['trading_longonly']
    print(f"  [Long-Short]  Return: {ls['total_return']:>+7.1%}  Sharpe: {ls['sharpe']:.2f}  MaxDD: {ls['max_drawdown']:>+7.1%}")
    print(f"  [Long-Only]   Return: {lo['total_return']:>+7.1%}  Sharpe: {lo['sharpe']:.2f}  MaxDD: {lo['max_drawdown']:>+7.1%}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 8: Save results
    # ------------------------------------------------------------------
    output = {
        'train_start': train_start,
        'train_end': train_end,
        'feature_names': feature_names,
        'raw_feature_names': raw_feature_names,
        'feature_mapping': feature_mapping,
        'regime_legend': REGIME_LEGEND,
        'rules_df': rules_df,
        'top_rules': rule_part,
        'linear_terms': linear_part,
        'oos_metrics': oos_metrics,
        'n_rules_total': len(rules_df),
        'n_rules_active': len(active_rules),
    }

    output_path = Path(f'data/processed/rulefit_regime_results{suffix}.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    logger.success(f"Results saved to {output_path}")
    logger.info(f"Total rules generated: {len(rules_df)}  |  Active rules: {len(active_rules)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
