#!/usr/bin/env python3
"""
Walk-Forward Cross-Validation with Knowledge Distillation

Implements knowledge distillation from LightGBM (teacher) to EBM (student)
using soft labels with temperature scaling.

Temperature Search:
- Searches T in {1, 2, 3, 4} during walk-forward validation
- For each fold and each T, trains a separate EBM distilled model
- Evaluates accuracy on validation set using hard labels
- Selects T with highest mean validation accuracy across all folds

Soft Labels with Temperature:
Given LightGBM predicted probabilities p for the positive class:
    logit = log(p / (1 - p + 1e-8))
    soft_pos = exp(logit / T) / (exp(logit / T) + exp(-logit / T))

EBM Distillation Training:
- EBM does not natively support soft labels
- Approximates soft label training via sample weighting:
    y_hard = (soft_pos >= 0.5).astype(int)
    sample_weight = |soft_pos - 0.5| * 2   (range 0 to 1)
- Trains EBM with fit(X_train, y_hard, sample_weight=sample_weight)

Results saved to data/processed/walk_forward_distillation_results.pkl
"""

import argparse
import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from interpret.glassbox import ExplainableBoostingClassifier

from src.ingestion.loader import DataLoader
from src.models import ModelTrainer, WalkForwardCV

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def compute_soft_labels(teacher_proba, temperature):
    """
    Compute soft labels from teacher probabilities using temperature scaling.
    
    Args:
        teacher_proba: Teacher predicted probabilities for positive class (array)
        temperature: Temperature parameter T (higher = softer)
        
    Returns:
        Soft labels (array of floats between 0 and 1)
    """
    # Compute logits from probabilities
    p = np.clip(teacher_proba, 1e-8, 1 - 1e-8)
    logits = np.log(p / (1 - p))
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Convert back to probabilities using sigmoid
    soft_labels = 1 / (1 + np.exp(-scaled_logits))
    
    return soft_labels


def train_ebm_distilled(X_train, soft_labels, config, fold_number, temperature):
    """
    Train EBM using soft labels via sample weighting.
    
    Args:
        X_train: Training features
        soft_labels: Soft labels from teacher (floats between 0 and 1)
        config: Configuration dictionary
        fold_number: Current fold number for logging
        temperature: Temperature used for soft labels
        
    Returns:
        Trained EBM model
    """
    # Convert soft labels to hard labels
    y_hard = (soft_labels >= 0.5).astype(int)
    
    # Compute sample weights: higher weight for more confident predictions
    # |soft_pos - 0.5| * 2 maps [0.5, 1.0] -> [0, 1]
    sample_weight = np.abs(soft_labels - 0.5) * 2
    
    logger.info(f"Training EBM distilled (T={temperature}, fold {fold_number})...")
    
    ebm_config = config.get('models', {}).get('ebm', {})
    
    model = ExplainableBoostingClassifier(
        max_bins=ebm_config.get('max_bins', 128),
        max_interaction_bins=ebm_config.get('max_interaction_bins', 32),
        interactions=ebm_config.get('interactions', 10),
        learning_rate=ebm_config.get('learning_rate', 0.01),
        max_rounds=ebm_config.get('max_rounds', 5000),
        min_samples_leaf=ebm_config.get('min_samples_leaf', 10),
        random_state=ebm_config.get('random_state', 42)
    )
    
    model.fit(X_train, y_hard, sample_weight=sample_weight)
    
    logger.success(f"EBM distilled (T={temperature}) trained for fold {fold_number}")
    
    return model


def train_ebm_threshold(X_train, teacher_proba, config, fold_number, threshold):
    """
    Train EBM filtering out ambiguous predictions via confidence thresholding.

    Keeps only training observations where:
        teacher_proba > threshold          → y_hard = 1 (long)
        teacher_proba < (1 - threshold)   → y_hard = 0 (short)
    Discards the ambiguous zone: (1-threshold) <= prob <= threshold.

    Sample weights = |prob - 0.5| * 2 applied to the kept subset.
    Validation is always performed on the FULL validation set (no filtering).

    Returns trained EBM, or None if too few samples remain after filtering.
    """
    n_total = len(teacher_proba)
    mask = (teacher_proba > threshold) | (teacher_proba < (1.0 - threshold))
    n_kept = int(mask.sum())
    n_discarded = n_total - n_kept
    logger.info(
        f"  thr={threshold:.2f}: kept {n_kept}/{n_total} "
        f"({100 * n_discarded / n_total:.1f}% discarded, {n_discarded} obs)"
    )

    if n_kept < 30:
        logger.warning(
            f"  thr={threshold:.2f}: only {n_kept} samples after filtering — skipped"
        )
        return None

    X_filtered = X_train.iloc[mask] if hasattr(X_train, 'iloc') else X_train[mask]
    proba_filtered = teacher_proba[mask]
    y_hard = (proba_filtered >= 0.5).astype(int)
    sample_weight = np.abs(proba_filtered - 0.5) * 2

    ebm_config = config.get('models', {}).get('ebm', {})
    model = ExplainableBoostingClassifier(
        max_bins=ebm_config.get('max_bins', 128),
        max_interaction_bins=ebm_config.get('max_interaction_bins', 32),
        interactions=ebm_config.get('interactions', 5),
        learning_rate=ebm_config.get('learning_rate', 0.01),
        max_rounds=ebm_config.get('max_rounds', 3000),
        min_samples_leaf=ebm_config.get('min_samples_leaf', 10),
        random_state=ebm_config.get('random_state', 42),
    )
    model.fit(X_filtered, y_hard, sample_weight=sample_weight)
    logger.success(f"  EBM (thr={threshold:.2f}) trained for fold {fold_number}")
    return model


def process_fold_thr(fold_data, config, trainer, thresholds):
    """
    Process a single fold for threshold search.

    Trains LightGBM once, then trains one EBM per threshold value using
    confidence filtering on teacher_proba.  Raw teacher probabilities
    (T=1, no temperature scaling) are used as the filter criterion — this
    is the cleanest confidence signal before any smoothing.

    Evaluates each EBM on the FULL validation set (no threshold applied
    at inference time).  Selection criterion: minimum Brier Score.

    Args:
        fold_data:   Fold dict from WalkForwardCV.split()
        config:      Configuration dictionary
        trainer:     ModelTrainer instance
        thresholds:  List of threshold values to search, e.g. [0.50, 0.55, 0.60]

    Returns:
        Dictionary with fold results including per-threshold metrics.
    """
    fold_number = fold_data['fold_number']

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PROCESSING FOLD {fold_number} - THRESHOLD SEARCH")
    logger.info("=" * 80)

    X_train = fold_data['X_train']
    y_train = fold_data['y_train']
    X_val = fold_data['X_val']
    y_val = fold_data['y_val']

    y_train_binary = (y_train == 1).astype(int)
    y_val_binary = (y_val == 1).astype(int)

    X_train_clean = X_train.reset_index(drop=True)
    X_val_clean = X_val.reset_index(drop=True)

    logger.info(f"Training samples: {len(X_train_clean)}")
    logger.info(f"Validation samples: {len(X_val_clean)}")
    logger.info(f"Features: {X_train_clean.shape[1]}")

    # Step 1: Train LightGBM teacher on hard labels
    logger.info("\n--- STEP 1: Training LightGBM Teacher ---")
    lightgbm_model = trainer._train_lightgbm(X_train_clean, y_train_binary, fold_number)
    if lightgbm_model is None:
        raise RuntimeError(f"LightGBM training failed for fold {fold_number} — cannot apply threshold distillation")

    lgbm_val_pred = lightgbm_model.predict(X_val_clean)
    lgbm_val_accuracy = accuracy_score(y_val_binary, lgbm_val_pred)
    logger.info(f"LightGBM teacher validation accuracy: {lgbm_val_accuracy:.4f}")

    # Raw teacher probabilities on training set (T=1: no temperature scaling)
    teacher_proba = lightgbm_model.predict_proba(X_train_clean)[:, 1]
    logger.info(
        f"Teacher proba: min={teacher_proba.min():.3f}  "
        f"max={teacher_proba.max():.3f}  mean={teacher_proba.mean():.3f}"
    )

    # Step 2: Train EBM for each threshold and evaluate on full validation set
    logger.info("\n--- STEP 2: Threshold Search ---")
    ebm_thr_models = {}
    thr_val_brier = {}
    thr_val_auc = {}
    thr_val_accuracy = {}
    thr_n_kept = {}

    for thr in thresholds:
        logger.info(f"\nThreshold thr={thr:.2f}:")

        ebm_model = train_ebm_threshold(
            X_train_clean, teacher_proba, config, fold_number, thr
        )

        if ebm_model is None:
            thr_val_brier[thr] = np.nan
            thr_val_auc[thr] = np.nan
            thr_val_accuracy[thr] = np.nan
            thr_n_kept[thr] = 0
            continue

        mask = (teacher_proba > thr) | (teacher_proba < (1.0 - thr))
        thr_n_kept[thr] = int(mask.sum())

        # Evaluate on FULL validation set
        ebm_val_pred = ebm_model.predict(X_val_clean)
        ebm_val_proba = ebm_model.predict_proba(X_val_clean)[:, 1]
        ebm_val_acc = accuracy_score(y_val_binary, ebm_val_pred)
        ebm_val_auc = roc_auc_score(y_val_binary, ebm_val_proba)
        ebm_val_brier = brier_score_loss(y_val_binary, ebm_val_proba)

        logger.info(
            f"  Acc={ebm_val_acc:.4f}  AUC={ebm_val_auc:.4f}  "
            f"Brier={ebm_val_brier:.4f}"
        )

        ebm_thr_models[thr] = ebm_model
        thr_val_brier[thr] = ebm_val_brier
        thr_val_auc[thr] = ebm_val_auc
        thr_val_accuracy[thr] = ebm_val_acc

    # Step 3: Log fold summary
    logger.info("\n--- FOLD SUMMARY (threshold search) ---")
    logger.info(f"LightGBM teacher: Acc={lgbm_val_accuracy:.4f}")
    for thr in thresholds:
        kept = thr_n_kept.get(thr, 0)
        pct_kept = 100 * kept / len(teacher_proba) if len(teacher_proba) > 0 else 0
        logger.info(
            f"thr={thr:.2f}: kept={kept} ({pct_kept:.0f}%)  "
            f"Brier={thr_val_brier.get(thr, float('nan')):.4f}  "
            f"AUC={thr_val_auc.get(thr, float('nan')):.4f}  "
            f"Acc={thr_val_accuracy.get(thr, float('nan')):.4f}"
        )

    return {
        'fold_number': fold_number,
        'train_start': fold_data['train_start'],
        'train_end': fold_data['train_end'],
        'val_start': fold_data['val_start'],
        'val_end': fold_data['val_end'],
        'lightgbm_model': lightgbm_model,
        'ebm_thr_models': ebm_thr_models,
        'lightgbm_val_accuracy': lgbm_val_accuracy,
        'thr_val_brier': thr_val_brier,
        'thr_val_auc': thr_val_auc,
        'thr_val_accuracy': thr_val_accuracy,
        'thr_n_kept': thr_n_kept,
        'X_val': X_val,
        'y_val': y_val,
        'y_val_binary': y_val_binary,
        'feature_names': fold_data['feature_names'],
    }


def process_fold(fold_data, config, trainer, temperatures):
    """
    Process a single fold: train teacher and distilled students.
    
    Args:
        fold_data: Fold data from WalkForwardCV
        config: Configuration dictionary
        trainer: ModelTrainer instance
        temperatures: List of temperatures to search
        
    Returns:
        Dictionary with fold results
    """
    fold_number = fold_data['fold_number']
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PROCESSING FOLD {fold_number} - KNOWLEDGE DISTILLATION")
    logger.info("=" * 80)
    
    X_train = fold_data['X_train']
    y_train = fold_data['y_train']
    X_val = fold_data['X_val']
    y_val = fold_data['y_val']
    
    # Convert labels from {-1, 1} to {0, 1}
    y_train_binary = (y_train == 1).astype(int)
    y_val_binary = (y_val == 1).astype(int)
    
    # Remove MultiIndex for model training
    X_train_clean = X_train.reset_index(drop=True)
    X_val_clean = X_val.reset_index(drop=True)
    
    logger.info(f"Training samples: {len(X_train_clean)}")
    logger.info(f"Validation samples: {len(X_val_clean)}")
    logger.info(f"Features: {X_train_clean.shape[1]}")
    
    # Step 1: Train LightGBM teacher on hard labels
    logger.info("\n--- STEP 1: Training LightGBM Teacher ---")
    lightgbm_model = trainer._train_lightgbm(X_train_clean, y_train_binary, fold_number)

    if lightgbm_model is None:
        raise RuntimeError(f"LightGBM training failed for fold {fold_number} — cannot distil")

    # Evaluate teacher on validation set
    lgbm_val_pred = lightgbm_model.predict(X_val_clean)
    lgbm_val_accuracy = accuracy_score(y_val_binary, lgbm_val_pred)
    logger.info(f"LightGBM teacher validation accuracy: {lgbm_val_accuracy:.4f}")
    
    # Step 2: Generate soft labels on training set for each temperature
    logger.info("\n--- STEP 2: Generating Soft Labels ---")
    teacher_proba = lightgbm_model.predict_proba(X_train_clean)[:, 1]
    
    # Step 3: Train EBM distilled for each temperature
    logger.info("\n--- STEP 3: Training EBM Distilled Models ---")
    ebm_distilled_models = {}
    distilled_val_accuracy = {}
    distilled_val_auc = {}

    for T in temperatures:
        logger.info(f"\nTemperature T={T}:")

        # Compute soft labels with this temperature
        soft_labels = compute_soft_labels(teacher_proba, T)
        logger.info(f"  Soft labels: min={soft_labels.min():.4f}, "
                   f"max={soft_labels.max():.4f}, mean={soft_labels.mean():.4f}")

        # Train EBM distilled
        ebm_model = train_ebm_distilled(
            X_train_clean, soft_labels, config, fold_number, T
        )

        # Evaluate on validation set
        ebm_val_pred = ebm_model.predict(X_val_clean)
        ebm_val_proba = ebm_model.predict_proba(X_val_clean)[:, 1]
        ebm_val_acc = accuracy_score(y_val_binary, ebm_val_pred)
        ebm_val_auc = roc_auc_score(y_val_binary, ebm_val_proba)

        logger.info(f"  EBM distilled (T={T}) — Accuracy: {ebm_val_acc:.4f}  AUC: {ebm_val_auc:.4f}")

        ebm_distilled_models[T] = ebm_model
        distilled_val_accuracy[T] = ebm_val_acc
        distilled_val_auc[T] = ebm_val_auc

    # Step 4: Log fold summary
    logger.info("\n--- FOLD SUMMARY ---")
    logger.info(f"LightGBM teacher:     {lgbm_val_accuracy:.4f}")
    for T in temperatures:
        logger.info(f"EBM distilled (T={T}):  Acc={distilled_val_accuracy[T]:.4f}  AUC={distilled_val_auc[T]:.4f}")
    
    return {
        'fold_number': fold_number,
        'train_start': fold_data['train_start'],
        'train_end': fold_data['train_end'],
        'val_start': fold_data['val_start'],
        'val_end': fold_data['val_end'],
        'lightgbm_model': lightgbm_model,
        'ebm_distilled_models': ebm_distilled_models,
        'lightgbm_val_accuracy': lgbm_val_accuracy,
        'distilled_val_accuracy': distilled_val_accuracy,
        'distilled_val_auc': distilled_val_auc,
        'X_val': X_val,
        'y_val': y_val,
        'y_val_binary': y_val_binary,
        'feature_names': fold_data['feature_names']
    }


def select_best_temperature(all_fold_results, temperatures):
    """
    Select best temperature based on mean validation AUC across folds.
    AUC is preferred over accuracy: threshold-independent and more robust with class imbalance.
    Returns (best_T, T_summary dict with mean/std for both AUC and accuracy).
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEMPERATURE SELECTION (criterion: mean validation AUC)")
    logger.info("=" * 80)

    T_summary = {}

    for T in temperatures:
        aucs = [fold['distilled_val_auc'][T] for fold in all_fold_results]
        accs = [fold['distilled_val_accuracy'][T] for fold in all_fold_results]
        T_summary[T] = {
            'mean_auc':  np.mean(aucs),
            'std_auc':   np.std(aucs),
            'aucs':      aucs,
            'mean_acc':  np.mean(accs),
            'std_acc':   np.std(accs),
        }

    # Select best temperature by mean AUC
    best_T = max(T_summary.keys(), key=lambda t: T_summary[t]['mean_auc'])

    logger.info(f"\n{'Temperature':<15} {'Mean AUC':<14} {'Std AUC':<12} {'Mean Acc':<12} {'Best'}")
    logger.info("-" * 65)
    for T in sorted(temperatures):
        s = T_summary[T]
        marker = " ← BEST" if T == best_T else ""
        logger.info(
            f"T={T:<13} {s['mean_auc']:<14.4f} {s['std_auc']:<12.4f} "
            f"{s['mean_acc']:<12.4f}{marker}"
        )
    logger.info("-" * 65)
    logger.info(f"Selected temperature: T={best_T}  (mean AUC={T_summary[best_T]['mean_auc']:.4f})")

    # Robustness check: report if AUC differences across temperatures are negligible.
    # When the AUC range is <0.005, temperature scaling has no meaningful effect.
    # This occurs because the EBM distillation approximates soft labels via sample
    # weighting (not true soft targets), so temperature only rescales weights — a
    # weaker signal than true label smoothing.  T=1 (no scaling) winning in this
    # regime is expected: it applies the highest confidence weights without
    # introducing noise from over-smoothing.
    auc_range = max(s['mean_auc'] for s in T_summary.values()) - min(s['mean_auc'] for s in T_summary.values())
    if auc_range < 0.005:
        logger.info(
            f"NOTE: AUC range across temperatures = {auc_range:.4f} < 0.005 — "
            "temperature scaling has negligible effect on this dataset. "
            "T=1 (hard labels) is functionally equivalent to T>1 for this distillation setup."
        )

    return best_T, T_summary


def select_best_threshold(all_fold_results, thresholds):
    """
    Select best confidence threshold based on mean validation Brier Score across folds.

    Brier Score is preferred over AUC for threshold selection: it measures
    probability calibration directly, which is the exact property that
    threshold filtering is designed to improve.  Lower is better.

    Returns (best_thr, thr_summary dict with mean/std for Brier, AUC, Accuracy).
    """
    logger.info("\n" + "=" * 80)
    logger.info("THRESHOLD SELECTION (criterion: mean validation Brier Score, lower = better)")
    logger.info("=" * 80)

    thr_summary = {}
    for thr in thresholds:
        briers = [fold['thr_val_brier'][thr] for fold in all_fold_results]
        aucs = [fold['thr_val_auc'][thr] for fold in all_fold_results]
        accs = [fold['thr_val_accuracy'][thr] for fold in all_fold_results]
        n_kept_list = [fold['thr_n_kept'].get(thr, 0) for fold in all_fold_results]
        thr_summary[thr] = {
            'mean_brier': float(np.nanmean(briers)),
            'std_brier':  float(np.nanstd(briers)),
            'mean_auc':   float(np.nanmean(aucs)),
            'mean_acc':   float(np.nanmean(accs)),
            'mean_n_kept': float(np.mean(n_kept_list)),
        }

    # Select by minimum mean Brier Score
    valid = {t: s for t, s in thr_summary.items() if not np.isnan(s['mean_brier'])}
    best_thr = min(valid.keys(), key=lambda t: valid[t]['mean_brier'])

    header = f"\n{'Threshold':<12} {'Mean Brier':<14} {'Std Brier':<12} {'Mean AUC':<12} {'Mean Kept':<12} {'Best'}"
    logger.info(header)
    logger.info("-" * 70)
    for thr in sorted(thresholds):
        s = thr_summary[thr]
        marker = " <- BEST" if thr == best_thr else ""
        logger.info(
            f"thr={thr:<8.2f} {s['mean_brier']:<14.4f} {s['std_brier']:<12.4f} "
            f"{s['mean_auc']:<12.4f} {s['mean_n_kept']:<12.0f}{marker}"
        )
    logger.info("-" * 70)
    logger.info(
        f"Selected threshold: {best_thr:.2f}  "
        f"(mean Brier={thr_summary[best_thr]['mean_brier']:.4f})"
    )

    brier_range = (
        max(s['mean_brier'] for s in valid.values())
        - min(s['mean_brier'] for s in valid.values())
    )
    if brier_range < 0.005:
        logger.info(
            f"NOTE: Brier range across thresholds = {brier_range:.4f} < 0.005 — "
            "confidence filtering has negligible effect on calibration for this dataset."
        )

    return best_thr, thr_summary


def main():
    """Run walk-forward cross-validation with knowledge distillation."""
    parser = argparse.ArgumentParser(description='Walk-forward CV with knowledge distillation')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Path to config YAML (default: config/config.yaml)')
    parser.add_argument(
        '--mode', default='temp', choices=['temp', 'thr'],
        help=(
            'temp (default): temperature search T={1,2,3,4}, saves _results{suffix}.pkl; '
            'thr: confidence threshold search [0.50,0.55,0.60], saves _results{suffix}_thr.pkl'
        ),
    )
    args = parser.parse_args()

    config_stem = Path(args.config).stem
    suffix = config_stem[len('config'):]

    logger.add(
        "logs/walk_forward_distillation_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )

    # Load configuration and data (common to both modes)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("\nLoading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()
    logger.info(f"Data loaded: {data.shape}")
    logger.info(
        f"Date range: {data.index.get_level_values('date').min()} to "
        f"{data.index.get_level_values('date').max()}"
    )

    wf_cv = WalkForwardCV(config)
    trainer = ModelTrainer(config)

    # ------------------------------------------------------------------
    # MODE: thr — confidence threshold search
    # ------------------------------------------------------------------
    if args.mode == 'thr':
        thresholds = [0.50, 0.55, 0.60]

        logger.info("=" * 80)
        logger.info("WALK-FORWARD CV — CONFIDENCE THRESHOLD SEARCH")
        logger.info("=" * 80)
        logger.info("Teacher: LightGBM (hard labels)")
        logger.info("Student: EBM (filtered by confidence threshold on teacher_proba)")
        logger.info(f"Thresholds: {thresholds}")
        logger.info("Selection criterion: minimum mean validation Brier Score")
        logger.info(f"Config: {args.config}  (output suffix: '{suffix}_thr')")
        logger.info("=" * 80)

        output_path = Path(f'data/processed/walk_forward_distillation_results{suffix}_thr.pkl')
        checkpoint_path = Path(f'data/processed/walk_forward_distillation_results{suffix}_thr_checkpoint.pkl')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_fold_results = []
        for fold_data in wf_cv.split(data):
            fold_results = process_fold_thr(fold_data, config, trainer, thresholds)
            all_fold_results.append(fold_results)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'all_fold_results': all_fold_results}, f)
            logger.debug(f"Checkpoint saved after fold {fold_results['fold_number']}")

        best_thr, thr_summary = select_best_threshold(all_fold_results, thresholds)

        # Also record best_T=1 (threshold mode always uses T=1 / raw proba)
        results = {
            'best_T': 1,
            'best_threshold': best_thr,
            'thr_summary': thr_summary,
            'all_fold_results': all_fold_results,
            'thresholds_searched': thresholds,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logger.success(f"Results saved to {output_path}")

        logger.info("\n" + "=" * 80)
        logger.info("THRESHOLD SEARCH COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total folds: {len(all_fold_results)}")
        logger.info(f"Thresholds searched: {thresholds}")
        logger.info(f"Best threshold: {best_thr:.2f}")
        logger.info(f"Best mean Brier Score: {thr_summary[best_thr]['mean_brier']:.4f}")
        logger.info(f"Results saved: {output_path}")
        logger.info("=" * 80)
        return

    # ------------------------------------------------------------------
    # MODE: temp (default) — temperature search (original behaviour)
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("WALK-FORWARD CROSS-VALIDATION WITH KNOWLEDGE DISTILLATION")
    logger.info("=" * 80)
    logger.info("Teacher: LightGBM (trained on hard labels)")
    logger.info("Student: EBM (trained on soft labels with temperature)")
    logger.info("Temperature search: T in {1, 2, 3, 4}")
    logger.info(f"Config: {args.config}  (output suffix: '{suffix}')")
    logger.info("=" * 80)

    temperatures = [1, 2, 3, 4]
    logger.info(f"\nTemperatures to search: {temperatures}")

    checkpoint_path = Path(f'data/processed/walk_forward_distillation_results{suffix}_checkpoint.pkl')
    all_fold_results = []

    for fold_data in wf_cv.split(data):
        fold_results = process_fold(fold_data, config, trainer, temperatures)
        all_fold_results.append(fold_results)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({'all_fold_results': all_fold_results}, f)
        logger.debug(f"Checkpoint saved after fold {fold_results['fold_number']}")

    best_T, T_summary = select_best_temperature(all_fold_results, temperatures)

    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    output_path = Path(f'data/processed/walk_forward_distillation_results{suffix}.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'best_T': best_T,
        'T_summary': T_summary,
        'all_fold_results': all_fold_results,
        'temperatures_searched': temperatures,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logger.success(f"Results saved to {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("KNOWLEDGE DISTILLATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total folds: {len(all_fold_results)}")
    logger.info(f"Temperatures searched: {temperatures}")
    logger.info(f"Best temperature: T={best_T}")
    logger.info(f"Best mean validation AUC: {T_summary[best_T]['mean_auc']:.4f}")
    logger.info(f"Results saved: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
