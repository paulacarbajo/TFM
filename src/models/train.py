"""
Model Training Module

Trains LightGBM and EBM models using walk-forward cross-validation.
RuleFit is excluded from walk-forward due to scalability constraints and is
applied separately on a representative sample for post-hoc rule extraction.
"""

from typing import Dict, Any, List, Optional
import time
import warnings

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.exceptions import ConvergenceWarning

# ML models
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from imodels import RuleFitClassifier

# SHAP for explainability
import shap

# Walk-forward CV
from .walk_forward import WalkForwardCV

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelTrainer:
    """
    Trains LightGBM and EBM models using walk-forward cross-validation.

    LightGBM serves as the primary high-capacity classifier. EBM is trained
    in parallel as a glassbox baseline, allowing direct comparison between a
    black-box ensemble and an inherently interpretable model under identical
    validation conditions.

    RuleFit is excluded from the walk-forward loop due to scalability
    constraints on large training sets. It is applied separately on a
    representative sample for post-hoc rule extraction and interpretability
    analysis.

    SHAP values are computed for LightGBM at each fold to track feature
    importance across the walk-forward period.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        lgbm_config (Dict): LightGBM hyperparameters from config
        ebm_config (Dict): EBM hyperparameters from config
        rulefit_config (Dict): RuleFit hyperparameters from config
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer.

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        models_config = config.get('models', {})

        self.lgbm_config = models_config.get('lightgbm', {})
        self.ebm_config = models_config.get('ebm', {})
        self.rulefit_config = models_config.get('rulefit', {})

        logger.info("ModelTrainer initialized")
        logger.info(f"LightGBM config: {self.lgbm_config}")
        logger.info(f"EBM config: {self.ebm_config}")

    def _convert_labels(self, y: pd.Series) -> np.ndarray:
        """
        Convert labels from {-1, 1} to {0, 1} for binary classification.

        LightGBM and EBM expect binary labels in {0, 1} format.
        The triple barrier produces labels in {-1, 1}, so conversion
        is required before training.

        Args:
            y: Series with labels (-1 or 1)

        Returns:
            Array with labels (0 or 1)
        """
        return (y == 1).astype(int).values

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        fold_number: int
    ) -> Optional[LGBMClassifier]:
        """
        Train LightGBM classifier on a single fold.

        Uses class_weight='balanced' to account for mild class imbalance
        between take-profit (label=1) and other outcomes (label=0 after
        conversion).

        Args:
            X_train: Training features (without MultiIndex)
            y_train: Training labels in {0, 1}
            fold_number: Current fold number for logging

        Returns:
            Trained LightGBM model, or None if training fails
        """
        try:
            logger.info(f"Training LightGBM fold {fold_number}...")
            start_time = time.time()

            model = LGBMClassifier(
                objective='binary',
                n_estimators=self.lgbm_config.get('n_estimators', 500),
                learning_rate=self.lgbm_config.get('learning_rate', 0.05),
                max_depth=self.lgbm_config.get('max_depth', 6),
                num_leaves=self.lgbm_config.get('num_leaves', 31),
                min_child_samples=self.lgbm_config.get('min_child_samples', 20),
                subsample=self.lgbm_config.get('subsample', 0.8),
                colsample_bytree=self.lgbm_config.get('colsample_bytree', 0.8),
                class_weight=self.lgbm_config.get('class_weight', 'balanced'),
                random_state=self.lgbm_config.get('random_state', 42),
                verbose=-1
            )

            model.fit(X_train, y_train)

            elapsed = time.time() - start_time
            logger.success(f"LightGBM fold {fold_number} trained in {elapsed:.2f}s")

            return model

        except Exception as e:
            logger.warning(f"LightGBM fold {fold_number} failed: {str(e)}")
            return None

    def _train_ebm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        fold_number: int
    ) -> Optional[ExplainableBoostingClassifier]:
        """
        Train Explainable Boosting Machine on a single fold.

        EBM is a generalised additive model trained with gradient boosting.
        Each feature contributes to the prediction through an individual shape
        function, making the model fully transparent while retaining ensemble-level
        accuracy. It is trained under identical conditions to LightGBM to allow
        direct performance comparison.

        Args:
            X_train: Training features (without MultiIndex)
            y_train: Training labels in {0, 1}
            fold_number: Current fold number for logging

        Returns:
            Trained EBM model, or None if training fails
        """
        try:
            logger.info(f"Training EBM fold {fold_number} (this may take 1-2 minutes)...")
            start_time = time.time()

            model = ExplainableBoostingClassifier(
                max_bins=self.ebm_config.get('max_bins', 256),
                max_interaction_bins=self.ebm_config.get('max_interaction_bins', 32),
                interactions=self.ebm_config.get('interactions', 10),
                learning_rate=self.ebm_config.get('learning_rate', 0.01),
                max_rounds=self.ebm_config.get('max_rounds', 5000),
                min_samples_leaf=self.ebm_config.get('min_samples_leaf', 2),
                random_state=self.ebm_config.get('random_state', 42)
            )

            model.fit(X_train, y_train)

            elapsed = time.time() - start_time
            logger.success(f"EBM fold {fold_number} trained in {elapsed:.2f}s")

            return model

        except Exception as e:
            logger.warning(f"EBM fold {fold_number} failed: {str(e)}")
            return None

    def _train_rulefit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        fold_number: int,
        feature_names: List[str]
    ) -> Optional[tuple[RuleFitClassifier, Dict[str, str]]]:
        """
        Train RuleFit classifier on a data sample.

        RuleFit generates IF-THEN rules from an ensemble of decision trees and
        combines them in a sparse linear model. It is used for post-hoc rule
        extraction and interpretability analysis, not for walk-forward evaluation.

        RuleFit does not scale well to large training sets. To keep training
        tractable, the input is subsampled to a maximum of 800 observations.
        Feature names with special characters are replaced with integer indices
        to avoid internal parsing errors, with a mapping preserved for readability.

        Args:
            X_train: Training features (without MultiIndex)
            y_train: Training labels in {0, 1}
            fold_number: Current fold number for logging
            feature_names: Original feature names for the mapping

        Returns:
            Tuple of (trained RuleFit model, feature name mapping),
            or None if training fails
        """
        try:
            logger.info(f"Training RuleFit fold {fold_number}...")
            start_time = time.time()

            model = RuleFitClassifier(
                tree_size=self.rulefit_config.get('tree_size', 4),
                max_rules=self.rulefit_config.get('max_rules', 500),
                random_state=self.rulefit_config.get('random_state', 42)
            )

            X_train_array = X_train.values
            simple_feature_names = [str(i) for i in range(X_train.shape[1])]

            feature_mapping = {
                simple: original
                for simple, original in zip(simple_feature_names, feature_names)
            }

            # RuleFit does not scale to large datasets — subsample if needed
            MAX_RULEFIT_SAMPLES = 800
            if X_train_array.shape[0] > MAX_RULEFIT_SAMPLES:
                logger.info(
                    f"RuleFit: subsampling from {X_train_array.shape[0]} "
                    f"to {MAX_RULEFIT_SAMPLES} samples"
                )
                idx = np.random.RandomState(42).choice(
                    X_train_array.shape[0],
                    MAX_RULEFIT_SAMPLES,
                    replace=False
                )
                X_train_array = X_train_array[idx]
                y_train = y_train[idx]

            model.fit(X_train_array, y_train, feature_names=simple_feature_names)

            elapsed = time.time() - start_time
            logger.success(f"RuleFit fold {fold_number} trained in {elapsed:.2f}s")

            return model, feature_mapping

        except Exception as e:
            logger.warning(f"RuleFit fold {fold_number} failed: {str(e)}")
            return None

    def _compute_shap_values(
        self,
        model: LGBMClassifier,
        X_val_clean: pd.DataFrame,
        fold_number: int
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        """
        Compute SHAP values for a trained LightGBM model.

        Uses TreeExplainer, which computes exact Shapley values by exploiting
        the tree structure — more efficient than the model-agnostic approximation.
        For binary classification, only the values for the positive class (label=1,
        take profit) are retained.

        Args:
            model: Trained LightGBM model
            X_val_clean: Validation features (without MultiIndex)
            fold_number: Current fold number for logging

        Returns:
            Tuple of (shap_values array, expected_value scalar),
            or (None, None) if computation fails
        """
        try:
            logger.info(f"Computing SHAP values for fold {fold_number}...")
            start_time = time.time()

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_clean)

            # For binary classification, retain values for the positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]

            elapsed = time.time() - start_time
            logger.success(f"SHAP values computed for fold {fold_number} in {elapsed:.2f}s")

            return shap_values, float(expected_value)

        except Exception as e:
            logger.warning(f"SHAP computation failed for fold {fold_number}: {str(e)}")
            return None, None

    def train_fold(self, fold_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all models on a single walk-forward fold.

        LightGBM and EBM are trained on the full training set.
        RuleFit is skipped in the walk-forward loop.
        SHAP values are computed for LightGBM on the validation set.

        The MultiIndex is removed before passing data to models, but preserved
        in X_val so that evaluate.py can align predictions with dates and tickers.

        Args:
            fold_data: Dictionary from WalkForwardCV.split()

        Returns:
            Dictionary with trained models, validation data, and SHAP values
        """
        fold_number = fold_data['fold_number']

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TRAINING MODELS - FOLD {fold_number}")
        logger.info("=" * 80)

        X_train = fold_data['X_train']
        y_train = fold_data['y_train']
        X_val = fold_data['X_val']
        y_val = fold_data['y_val']
        feature_names = fold_data['feature_names']

        y_train_binary = self._convert_labels(y_train)
        y_val_binary = self._convert_labels(y_val)

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Features: {len(feature_names)}")

        # Remove MultiIndex for model training
        # X_val with MultiIndex is preserved for evaluate.py
        X_train_clean = X_train.reset_index(drop=True)
        X_val_clean = X_val.reset_index(drop=True)

        training_times = {}
        models = {}

        # LightGBM
        start_time = time.time()
        lgbm_model = self._train_lightgbm(X_train_clean, y_train_binary, fold_number)
        training_times['lightgbm'] = time.time() - start_time
        models['lightgbm'] = lgbm_model

        # EBM
        start_time = time.time()
        ebm_model = self._train_ebm(X_train_clean, y_train_binary, fold_number)
        training_times['ebm'] = time.time() - start_time
        models['ebm'] = ebm_model

        # RuleFit — skipped in walk-forward, used separately for interpretability
        training_times['rulefit'] = 0
        models['rulefit'] = None
        logger.info("RuleFit: skipped in walk-forward (applied separately for rule extraction)")

        # SHAP values for LightGBM
        shap_values = None
        shap_expected_value = None
        if lgbm_model is not None:
            shap_values, shap_expected_value = self._compute_shap_values(
                lgbm_model, X_val_clean, fold_number
            )

        logger.info("")
        logger.info("Training summary:")
        logger.info(f"  LightGBM: {'SUCCESS' if lgbm_model else 'FAILED'} "
                    f"({training_times['lightgbm']:.2f}s)")
        logger.info(f"  EBM:      {'SUCCESS' if ebm_model else 'FAILED'} "
                    f"({training_times['ebm']:.2f}s)")
        logger.info(f"  RuleFit:  SKIPPED ({training_times['rulefit']:.2f}s)")
        logger.info(f"Total training time: {sum(training_times.values()):.2f}s")

        return {
            'fold_number': fold_number,
            'train_start': fold_data['train_start'],
            'train_end': fold_data['train_end'],
            'val_start': fold_data['val_start'],
            'val_end': fold_data['val_end'],
            'models': models,
            'X_val': X_val,
            'X_val_clean': X_val_clean,
            'y_val': y_val,
            'y_val_binary': y_val_binary,
            'val_dates': fold_data['val_dates'],
            'feature_names': feature_names,
            'training_time': training_times,
            'shap_values': shap_values,
            'shap_expected_value': shap_expected_value
        }

    def train_all_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Train models on all walk-forward folds.

        Args:
            data: DataFrame with MultiIndex (ticker, date) and all feature columns

        Returns:
            List of result dictionaries, one per fold
        """
        logger.info("=" * 80)
        logger.info("WALK-FORWARD MODEL TRAINING")
        logger.info("=" * 80)

        wf_cv = WalkForwardCV(self.config)

        results = []
        for fold_data in wf_cv.split(data):
            fold_number = fold_data['fold_number']
            logger.info(f"\nTraining fold {fold_number}...")
            results.append(self.train_fold(fold_data))

        logger.info("")
        logger.info("=" * 80)
        logger.info("WALK-FORWARD TRAINING COMPLETE")
        logger.info(f"Trained {len(results)} folds successfully")
        logger.info("=" * 80)

        return results
