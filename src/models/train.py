"""
Model Training Module
======================

Provides ModelTrainer, which trains classifiers within the walk-forward CV framework.

Three-tier architecture:
    Tier 1 — LightGBM (teacher): trained per fold in train_fold(). SHAP computed on val set.
    Tier 2 — EBM Distilled: trained in run_walk_forward_distillation.py via knowledge
             distillation (LightGBM soft-label probabilities replace hard labels).
    Tier 3 — RuleFit: trained in run_rulefit_distillation.py on a pooled OOS sample.

_train_lightgbm, _train_ebm, _train_rulefit are reusable building blocks.
train_fold() calls only _train_lightgbm and _compute_shap_values.
"""

from typing import Dict, Any, List, Optional
import time
import warnings

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from imodels import RuleFitClassifier
import shap

from .walk_forward import WalkForwardCV

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelTrainer:
    """
    Trains classifiers for the walk-forward pipeline.
    train_fold() trains only LightGBM; EBM and RuleFit are helpers for distillation scripts.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        models_config = config.get('models', {})

        self.lgbm_config   = models_config.get('lightgbm', {})
        self.ebm_config    = models_config.get('ebm', {})
        self.rulefit_config = models_config.get('rulefit', {})

        logger.info("ModelTrainer initialised")
        logger.info(f"LightGBM config: {self.lgbm_config}")
        logger.info(
            "EBM and RuleFit configs loaded (used by distillation / "
            "rule-extraction scripts, not by train_fold)"
        )

    # ------------------------------------------------------------------
    # Label conversion
    # ------------------------------------------------------------------

    def _convert_labels(self, y: pd.Series) -> np.ndarray:
        """Convert label_binary {−1, +1} → {0, 1} for LightGBM/EBM (binary objective)."""
        return (y == 1).astype(int).values

    # ------------------------------------------------------------------
    # Model builders (reusable across walk-forward and distillation)
    # ------------------------------------------------------------------

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        fold_number: int
    ) -> Optional[LGBMClassifier]:
        """
        Train a LightGBM binary classifier on one walk-forward fold.

        ~724 training samples per fold (3-year rolling window, SPY daily), 10 features
        (14 with regime columns). Parameters tuned for regularisation on small tabular data:
        n_estimators=300 (no early stopping — val set must not influence training),
        num_leaves=20 (<<2^max_depth=64, limits variance), min_child_samples=50 (~7% of fold),
        subsample/colsample=0.8 (stochastic boosting), reg_lambda=1.0 (L2), class_weight='balanced'.
        """
        try:
            logger.info(f"  Training LightGBM (fold {fold_number})...")
            start = time.time()

            model = LGBMClassifier(
                objective='binary',
                n_estimators=self.lgbm_config.get('n_estimators', 300),
                learning_rate=self.lgbm_config.get('learning_rate', 0.05),
                max_depth=self.lgbm_config.get('max_depth', 6),
                num_leaves=self.lgbm_config.get('num_leaves', 20),
                min_child_samples=self.lgbm_config.get('min_child_samples', 50),
                subsample=self.lgbm_config.get('subsample', 0.8),
                colsample_bytree=self.lgbm_config.get('colsample_bytree', 0.8),
                reg_alpha=self.lgbm_config.get('reg_alpha', 0.1),
                reg_lambda=self.lgbm_config.get('reg_lambda', 1.0),
                min_gain_to_split=self.lgbm_config.get('min_gain_to_split', 0.01),
                class_weight=self.lgbm_config.get('class_weight', 'balanced'),
                random_state=self.lgbm_config.get('random_state', 42),
                verbose=-1,
            )

            model.fit(X_train, y_train)

            logger.success(
                f"  LightGBM fold {fold_number} trained  |  "
                f"{time.time() - start:.2f}s"
            )
            return model

        except Exception as e:
            logger.warning(f"  LightGBM fold {fold_number} failed: {e}")
            return None

    def _train_ebm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        fold_number: int
    ) -> Optional[ExplainableBoostingClassifier]:
        """
        Train an EBM (GAM with pairwise interactions) on one fold. NOT called by train_fold.
        Called by run_walk_forward_distillation.py with soft labels from LightGBM.

        max_bins=128 (~5–6 samples/bin on ~724 fold rows), interactions=5,
        max_rounds=3000 with lr=0.01, min_samples_leaf=10.
        y_train accepts hard labels {0,1} or soft labels [0–1] (distillation).
        """
        try:
            logger.info(f"  Training EBM (fold {fold_number}, ~1–2 min)...")
            start = time.time()

            model = ExplainableBoostingClassifier(
                max_bins=self.ebm_config.get('max_bins', 128),
                max_interaction_bins=self.ebm_config.get('max_interaction_bins', 32),
                interactions=self.ebm_config.get('interactions', 10),
                learning_rate=self.ebm_config.get('learning_rate', 0.01),
                max_rounds=self.ebm_config.get('max_rounds', 5000),
                min_samples_leaf=self.ebm_config.get('min_samples_leaf', 10),
                random_state=self.ebm_config.get('random_state', 42),
            )

            model.fit(X_train, y_train)

            logger.success(
                f"  EBM fold {fold_number} trained  |  "
                f"{time.time() - start:.2f}s"
            )
            return model

        except Exception as e:
            logger.warning(f"  EBM fold {fold_number} failed: {e}")
            return None

    def _train_rulefit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        fold_number: int,
        feature_names: List[str],
        sample_weight: Optional[np.ndarray] = None
    ) -> Optional[tuple]:
        """
        Train a RuleFit classifier on a stratified subsample (max 800 rows — scalability limit).
        NOT called by train_fold. Called by distillation/rule-extraction scripts.

        Features are renamed to integer strings during fit (imodels parses names and chokes on
        underscores); a feature_mapping dict is returned to restore readable names downstream.
        Features are StandardScaler-normalised so linear-term coefficients are comparable.
        sample_weight is ignored (imodels does not support it); soft-label signal goes via y_train.

        Returns (RuleFitClassifier, feature_mapping, scaler) or None if training fails.
        """
        try:
            logger.info(f"  Training RuleFit (fold {fold_number})...")
            start = time.time()

            model = RuleFitClassifier(
                n_estimators=self.rulefit_config.get('n_estimators', 100),
                tree_size=self.rulefit_config.get('tree_size', 3),
                max_rules=self.rulefit_config.get('max_rules', 50),
                alpha=self.rulefit_config.get('alpha', 0.1),
                include_linear=self.rulefit_config.get('include_linear', True),
                random_state=self.rulefit_config.get('random_state', 42),
            )

            X_arr = X_train.values
            simple_names = [str(i) for i in range(X_train.shape[1])]
            feature_mapping = dict(zip(simple_names, feature_names))

            # Standardise features so linear-term coefficients are on a common
            # scale (units: standard deviations).  Without this, features with
            # small magnitude (e.g. vol_20d ≈ 0.01) get inflated coefficients
            # and their signs become unstable due to multicollinearity.
            scaler = StandardScaler()
            X_arr = scaler.fit_transform(X_arr)

            # Subsample if needed (scalability constraint)
            MAX_SAMPLES = 800
            if X_arr.shape[0] > MAX_SAMPLES:
                logger.info(
                    f"  RuleFit: subsampling {X_arr.shape[0]} → {MAX_SAMPLES} "
                    "(stratified)"
                )
                try:
                    sss = StratifiedShuffleSplit(
                        n_splits=1, train_size=MAX_SAMPLES, random_state=42
                    )
                    idx = next(sss.split(X_arr, y_train))[0]
                except ValueError as e:
                    logger.warning(f"  StratifiedShuffleSplit failed ({e}), random fallback")
                    idx = np.random.RandomState(42).choice(
                        X_arr.shape[0], MAX_SAMPLES, replace=False
                    )
                X_arr   = X_arr[idx]
                y_train = y_train[idx]
                if sample_weight is not None:
                    sample_weight = sample_weight[idx]

            if sample_weight is not None:
                # sample_weight not supported by imodels — confidence signal carried via y_train soft labels
                logger.info("  RuleFit: sample_weight ignored (not supported by imodels)")

            model.fit(X_arr, y_train, feature_names=simple_names)

            logger.success(
                f"  RuleFit fold {fold_number} trained  |  "
                f"{time.time() - start:.2f}s"
            )
            return model, feature_mapping, scaler

        except Exception as e:
            logger.warning(f"  RuleFit fold {fold_number} failed: {e}")
            return None

    def _compute_shap_values(
        self,
        model: LGBMClassifier,
        X_val_clean: pd.DataFrame,
        fold_number: int
    ) -> tuple:
        """
        Compute exact SHAP values (TreeExplainer) on the validation set.
        Returns positive-class values only (take_profit, label=1).
        Returns (None, None) if computation fails.
        """
        try:
            logger.info(f"  Computing SHAP values (fold {fold_number})...")
            start = time.time()

            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_val_clean)

            # Retain positive-class values only
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                exp_val = exp_val[1]

            logger.success(
                f"  SHAP fold {fold_number} done  |  "
                f"{time.time() - start:.2f}s"
            )
            return shap_vals, float(exp_val)

        except Exception as e:
            logger.warning(f"  SHAP fold {fold_number} failed: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Walk-forward orchestration
    # ------------------------------------------------------------------

    def train_fold(self, fold_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train LightGBM + SHAP for one walk-forward fold. EBM and RuleFit are skipped here.
        MultiIndex is stripped for training but preserved in X_val for backtest date alignment.
        """
        fold_number = fold_data['fold_number']

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"FOLD {fold_number} — TRAINING")
        logger.info("=" * 80)

        X_train      = fold_data['X_train']
        y_train      = fold_data['y_train']
        X_val        = fold_data['X_val']
        y_val        = fold_data['y_val']
        feature_names = fold_data['feature_names']

        y_train_binary = self._convert_labels(y_train)
        y_val_binary   = self._convert_labels(y_val)

        logger.info(
            f"Train: {len(X_train)} samples  |  "
            f"Val: {len(X_val)} samples  |  "
            f"Features: {len(feature_names)}"
        )
        logger.info(
            f"Train period: {fold_data['train_start'].date()} → "
            f"{fold_data['train_end'].date()}"
        )
        logger.info(
            f"Val period:   {fold_data['val_start'].date()} → "
            f"{fold_data['val_end'].date()}"
        )

        # Strip MultiIndex for sklearn/LightGBM; preserve X_val for backtest alignment
        X_train_clean = X_train.reset_index(drop=True)
        X_val_clean   = X_val.reset_index(drop=True)

        training_times: Dict[str, float] = {}
        models: Dict[str, Any] = {}

        # --- Tier 1: LightGBM (teacher) ---
        start = time.time()
        lgbm_model = self._train_lightgbm(X_train_clean, y_train_binary, fold_number)
        training_times['lightgbm'] = time.time() - start
        models['lightgbm'] = lgbm_model

        # --- Tier 2: EBM Distilled — trained separately ---
        models['ebm'] = None
        training_times['ebm'] = 0.0

        # --- Tier 3: RuleFit — trained separately ---
        models['rulefit'] = None
        training_times['rulefit'] = 0.0

        # --- SHAP for LightGBM ---
        shap_values         = None
        shap_expected_value = None
        if lgbm_model is not None:
            shap_values, shap_expected_value = self._compute_shap_values(
                lgbm_model, X_val_clean, fold_number
            )

        logger.info("")
        logger.info("Fold training summary:")
        logger.info(
            f"  LightGBM: {'OK' if lgbm_model else 'FAILED'}  "
            f"({training_times['lightgbm']:.2f}s)"
        )
        logger.info("  EBM:      SKIPPED → run_walk_forward_distillation.py")
        logger.info("  RuleFit:  SKIPPED → run_rulefit_distillation.py")
        logger.info(f"  SHAP:     {'OK' if shap_values is not None else 'FAILED'}")

        return {
            'fold_number':        fold_number,
            'train_start':        fold_data['train_start'],
            'train_end':          fold_data['train_end'],
            'val_start':          fold_data['val_start'],
            'val_end':            fold_data['val_end'],
            'models':             models,
            'X_val':              X_val,           # MultiIndex preserved for backtest
            'X_val_clean':        X_val_clean,     # MultiIndex stripped for scoring
            'y_val':              y_val,
            'y_val_binary':       y_val_binary,
            'val_dates':          fold_data['val_dates'],
            'feature_names':      feature_names,
            'training_time':      training_times,
            'shap_values':        shap_values,
            'shap_expected_value': shap_expected_value,
        }

    def train_all_folds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run the full walk-forward loop and return a list of fold result dicts."""
        logger.info("=" * 80)
        logger.info("WALK-FORWARD TRAINING — ALL FOLDS")
        logger.info("=" * 80)

        wf_cv   = WalkForwardCV(self.config)
        results = []

        for fold_data in wf_cv.split(data):
            results.append(self.train_fold(fold_data))

        logger.info("")
        logger.info("=" * 80)
        logger.success(f"WALK-FORWARD COMPLETE  |  {len(results)} folds")
        logger.info("=" * 80)

        return results
