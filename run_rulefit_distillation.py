#!/usr/bin/env python3
"""
RuleFit Distillation — Interpretable Rules from EBM Distilled

Implements the final step of the two-stage distillation chain:
    LightGBM (teacher)
        → EBM Distilled (student, trained on LightGBM soft labels)
            → RuleFit (trained on EBM Distilled soft labels → explicit if-then rules)

RuleFit converts the soft knowledge captured by EBM Distilled into human-readable
trading rules of the form:
    IF rsi_14 > 68.5 AND vol_20d < 0.010  →  LONG  (coef=+0.31, support=18%)
    IF macd_hist < -0.0008 AND bb_pct < 0.21  →  SHORT  (coef=-0.22, support=12%)

Training data: IS last fold (e.g. 2016-2019 for 2008 baseline).
Evaluation: full OOS period 2020-2024.

Usage:
    python run_rulefit_distillation.py
    python run_rulefit_distillation.py --config config/config_2010.yaml
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

from src.ingestion.loader import DataLoader
from src.models.train import ModelTrainer

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)



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
    parser = argparse.ArgumentParser(description='RuleFit distillation from EBM Distilled soft labels')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to config YAML (default: config/config.yaml)')
    args = parser.parse_args()

    config_stem = Path(args.config).stem
    suffix = config_stem[len('config'):]

    logger.add(
        "logs/rulefit_distillation_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    logger.info("=" * 80)
    logger.info("RULEFIT DISTILLATION: INTERPRETABLE RULES FROM EBM DISTILLED")
    logger.info("=" * 80)
    logger.info("Chain: LightGBM → EBM Distilled → RuleFit")
    logger.info(f"Config: {args.config}  (suffix: '{suffix}')")
    logger.info("=" * 80)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Step 1: Load distillation PKL — get last IS fold's EBM Distilled
    # ------------------------------------------------------------------
    distill_path = Path(f'data/processed/walk_forward_distillation_results{suffix}.pkl')
    if not distill_path.exists():
        raise FileNotFoundError(
            f"Distillation results not found: {distill_path}\n"
            "Run run_walk_forward_distillation.py first."
        )

    with open(distill_path, 'rb') as f:
        distill = pickle.load(f)

    best_T = distill['best_T']
    last_fold = distill['all_fold_results'][-1]
    ebm_distilled = last_fold['ebm_distilled_models'][best_T]
    train_start = last_fold['train_start']
    train_end = last_fold['train_end']
    feature_names = last_fold['feature_names']
    fold_number = last_fold['fold_number']

    logger.info(f"Best temperature: T={best_T}")
    logger.info(f"IS last fold: {fold_number}  ({train_start.date()} → {train_end.date()})")
    logger.info(f"Features ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # Step 2: Load data and recreate IS last fold training split
    # ------------------------------------------------------------------
    logger.info("\nLoading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()

    # Extract SPY (single ticker)
    if 'ticker' in data.index.names:
        spy = data.xs('SPY', level='ticker')
    else:
        spy = data

    # Extend window by 1 fold for more temporally proximate training data (2015-2019 vs 2016-2019)
    extended_train_start = train_start - pd.DateOffset(years=1)
    train_data = spy[
        (spy.index >= extended_train_start) &
        (spy.index < train_end)
    ].copy()
    logger.info(f"Extended IS window: {extended_train_start.date()} -> {train_end.date()} (original fold start: {train_start.date()})")

    valid = (
        train_data['label_binary'].notna() &
        train_data[feature_names].notna().all(axis=1)
    )
    train_clean = train_data[valid]
    X_train = train_clean[feature_names].values

    logger.info(f"Training samples: {len(X_train)}  (limit: 800 — {'will subsample' if len(X_train) > 800 else 'OK'})")

    # ------------------------------------------------------------------
    # Step 3: Generate soft labels from EBM Distilled
    # ------------------------------------------------------------------
    logger.info("\nGenerating soft labels from EBM Distilled...")

    ebm_proba = ebm_distilled.predict_proba(X_train)[:, 1]

    # Temperature scaling on EBM Distilled probabilities
    p = np.clip(ebm_proba, 1e-8, 1 - 1e-8)
    logits = np.log(p / (1 - p))
    soft_pos = 1 / (1 + np.exp(-logits / best_T))

    y_hard = (soft_pos >= 0.5).astype(int)
    sample_weight = np.abs(soft_pos - 0.5) * 2  # confidence: 0=uncertain, 1=certain

    n_pos = y_hard.sum()
    n_neg = len(y_hard) - n_pos
    logger.info(f"Soft labels: min={soft_pos.min():.3f}  max={soft_pos.max():.3f}  mean={soft_pos.mean():.3f}")
    logger.info(f"Hard labels: {n_pos} positive ({n_pos/len(y_hard)*100:.1f}%), {n_neg} negative")
    logger.info(f"Sample weights: min={sample_weight.min():.3f}  max={sample_weight.max():.3f}  mean={sample_weight.mean():.3f}")

    # ------------------------------------------------------------------
    # Step 4: Train RuleFit
    # ------------------------------------------------------------------
    logger.info("\nTraining RuleFit on EBM Distilled soft labels...")

    trainer = ModelTrainer(config)
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    result = trainer._train_rulefit(
        X_train_df, y_hard, fold_number, feature_names, sample_weight=sample_weight
    )
    if result is None:
        raise RuntimeError("RuleFit training failed — check logs for details.")

    rulefit_model, feature_mapping = result
    logger.success("RuleFit trained successfully.")

    # ------------------------------------------------------------------
    # Step 5: Extract and display readable rules
    # ------------------------------------------------------------------
    logger.info("\nExtracting rules...")

    # _get_rules() returns DataFrame with columns: rule, type, coef, support, importance
    # Rules use integer indices ('0', '1', ...) — translate back to original feature names
    rules_df = rulefit_model._get_rules().copy()

    def translate_rule(rule_str, mapping):
        """Replace integer feature indices with original names — single-pass to avoid cascade."""
        sorted_items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
        pattern = '|'.join(re.escape(idx) for idx, _ in sorted_items)
        full_pattern = rf'(?<![.\d])({pattern})(?![.\d])'
        return re.sub(full_pattern, lambda m: mapping[m.group(0)], rule_str)

    rules_df['rule'] = rules_df['rule'].apply(lambda r: translate_rule(str(r), feature_mapping))

    # Filter meaningful rules
    active_rules = rules_df[
        (rules_df['coef'].abs() > 0) &
        (rules_df['support'] > 0.01)
    ].copy()
    active_rules = active_rules.sort_values('coef', key=abs, ascending=False)

    # Separate if-then rules from linear terms
    is_rule = active_rules['type'] == 'rule'
    rule_part = active_rules[is_rule].head(20)
    linear_part = active_rules[~is_rule].head(15)

    print("\n" + "=" * 80)
    print("TOP IF-THEN RULES (sorted by |coefficient|)")
    print("=" * 80)
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
    # Step 6: Evaluate on OOS period (2020-2024)
    # ------------------------------------------------------------------
    logger.info("\nEvaluating RuleFit on OOS period 2020-2024...")

    oos_data = spy[
        (spy.index >= '2020-01-01') &
        (spy.index <= '2024-12-31')
    ].copy()

    oos_valid = (
        oos_data['label_binary'].notna() &
        oos_data['ret_1d_forward'].notna() &
        oos_data[feature_names].notna().all(axis=1)
    )
    oos_clean = oos_data[oos_valid]
    X_oos = oos_clean[feature_names].values
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
    # Step 7: Save results
    # ------------------------------------------------------------------
    output = {
        'best_T': best_T,
        'train_start': train_start,
        'train_end': train_end,
        'feature_names': feature_names,
        'feature_mapping': feature_mapping,
        'rules_df': rules_df,
        'top_rules': rule_part,
        'linear_terms': linear_part,
        'oos_metrics': oos_metrics,
        'n_rules_total': len(rules_df),
        'n_rules_active': len(active_rules),
    }

    output_path = Path(f'data/processed/rulefit_distillation_results{suffix}.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

    logger.success(f"Results saved to {output_path}")
    logger.info(f"Total rules generated: {len(rules_df)}  |  Active rules: {len(active_rules)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
