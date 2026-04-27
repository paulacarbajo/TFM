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

import warnings
import yaml
import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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
        max_bins=ebm_config.get('max_bins', 256),
        max_interaction_bins=ebm_config.get('max_interaction_bins', 32),
        interactions=ebm_config.get('interactions', 10),
        learning_rate=ebm_config.get('learning_rate', 0.01),
        max_rounds=ebm_config.get('max_rounds', 5000),
        min_samples_leaf=ebm_config.get('min_samples_leaf', 2),
        random_state=ebm_config.get('random_state', 42)
    )
    
    model.fit(X_train, y_hard, sample_weight=sample_weight)
    
    logger.success(f"EBM distilled (T={temperature}) trained for fold {fold_number}")
    
    return model


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
        
        # Evaluate on validation set using hard labels
        ebm_val_pred = ebm_model.predict(X_val_clean)
        ebm_val_acc = accuracy_score(y_val_binary, ebm_val_pred)
        
        logger.info(f"  EBM distilled (T={T}) validation accuracy: {ebm_val_acc:.4f}")
        
        ebm_distilled_models[T] = ebm_model
        distilled_val_accuracy[T] = ebm_val_acc
    
    # Step 4: Log fold summary
    logger.info("\n--- FOLD SUMMARY ---")
    logger.info(f"LightGBM teacher:     {lgbm_val_accuracy:.4f}")
    for T in temperatures:
        logger.info(f"EBM distilled (T={T}):  {distilled_val_accuracy[T]:.4f}")
    
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
        'X_val': X_val,
        'y_val': y_val,
        'y_val_binary': y_val_binary,
        'feature_names': fold_data['feature_names']
    }


def select_best_temperature(all_fold_results, temperatures):
    """
    Select best temperature based on mean validation accuracy across folds.
    
    Args:
        all_fold_results: List of fold result dictionaries
        temperatures: List of temperatures searched
        
    Returns:
        Tuple of (best_T, T_accuracy_summary dict)
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEMPERATURE SELECTION")
    logger.info("=" * 80)
    
    # Compute mean accuracy per temperature
    T_accuracy_summary = {}
    
    for T in temperatures:
        accuracies = [fold['distilled_val_accuracy'][T] for fold in all_fold_results]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        T_accuracy_summary[T] = {
            'mean': mean_acc,
            'std': std_acc,
            'accuracies': accuracies
        }
    
    # Select best temperature
    best_T = max(T_accuracy_summary.keys(), key=lambda t: T_accuracy_summary[t]['mean'])
    
    # Log summary table
    logger.info("\nValidation Accuracy by Temperature (mean ± std across folds):")
    logger.info("-" * 80)
    logger.info(f"{'Temperature':<15} {'Mean Accuracy':<20} {'Std':<15} {'Best':<10}")
    logger.info("-" * 80)
    
    for T in sorted(temperatures):
        summary = T_accuracy_summary[T]
        is_best = " ← BEST" if T == best_T else ""
        logger.info(
            f"T={T:<13} {summary['mean']:<20.4f} {summary['std']:<15.4f} {is_best}"
        )
    
    logger.info("-" * 80)
    logger.info(f"\nSelected temperature: T={best_T}")
    logger.info(f"Mean validation accuracy: {T_accuracy_summary[best_T]['mean']:.4f}")
    
    return best_T, T_accuracy_summary


def main():
    """Run walk-forward cross-validation with knowledge distillation."""
    logger.add(
        "logs/walk_forward_distillation_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    logger.info("=" * 80)
    logger.info("WALK-FORWARD CROSS-VALIDATION WITH KNOWLEDGE DISTILLATION")
    logger.info("=" * 80)
    logger.info("Teacher: LightGBM (trained on hard labels)")
    logger.info("Student: EBM (trained on soft labels with temperature)")
    logger.info("Temperature search: T in {1, 2, 3, 4}")
    logger.info("=" * 80)
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info("\nLoading data...")
    loader = DataLoader(config)
    data = loader.load_engineered_features()
    
    logger.info(f"Data loaded: {data.shape}")
    logger.info(
        f"Date range: {data.index.get_level_values('date').min()} to "
        f"{data.index.get_level_values('date').max()}"
    )
    
    # Initialize components
    wf_cv = WalkForwardCV(config)
    trainer = ModelTrainer(config)
    
    # Temperature search space
    temperatures = [1, 2, 3, 4]
    logger.info(f"\nTemperatures to search: {temperatures}")
    
    # Process all folds
    all_fold_results = []
    
    for fold_data in wf_cv.split(data):
        fold_results = process_fold(fold_data, config, trainer, temperatures)
        all_fold_results.append(fold_results)
    
    # Select best temperature
    best_T, T_accuracy_summary = select_best_temperature(all_fold_results, temperatures)
    
    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    output_path = Path('data/processed/walk_forward_distillation_results.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'best_T': best_T,
        'T_accuracy_summary': T_accuracy_summary,
        'all_fold_results': all_fold_results,
        'temperatures_searched': temperatures
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.success(f"Results saved to {output_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("KNOWLEDGE DISTILLATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total folds: {len(all_fold_results)}")
    logger.info(f"Temperatures searched: {temperatures}")
    logger.info(f"Best temperature: T={best_T}")
    logger.info(f"Best mean validation accuracy: {T_accuracy_summary[best_T]['mean']:.4f}")
    logger.info(f"Results saved: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
