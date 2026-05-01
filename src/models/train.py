"""
Model Training Module
======================

Provides ``ModelTrainer``, which trains and evaluates classifiers within the
walk-forward cross-validation framework.

Three-tier model architecture
-------------------------------
The pipeline uses three model tiers, each with a different role and a
different script that orchestrates it:

    Tier 1 — LightGBM (teacher)
        Trained per fold in ``train_fold()``.  High-capacity gradient-boosted
        ensemble.  Primary metric is AUC on the fold validation set.
        SHAP values are computed at every fold for feature importance tracking.
        Orchestrated by: ``run_walk_forward.py``

    Tier 2 — EBM Distilled (interpretable student)
        NOT trained in ``train_fold()``.  Trained separately in
        ``run_walk_forward_distillation.py`` using knowledge distillation:
        LightGBM soft-label probabilities (temperature T∈{1,2,3,4}) replace
        hard ``label_binary`` targets, and the EBM is trained with
        ``sample_weight`` proportional to the soft-label entropy.
        Rationale: training EBM directly on hard labels limits its AUC to that
        of the label noise floor; distilling from LightGBM's smoother
        probability surface improves EBM generalisation while retaining full
        interpretability.

    Tier 3 — RuleFit (rule extractor)
        NOT trained in ``train_fold()``.  Trained separately in
        ``run_rulefit_distillation.py`` on a pooled OOS sample using the same
        distillation soft labels.  Generates IF-THEN trading rules that are
        human-readable and directly actionable.
        Scalability note: RuleFit does not scale to fold-sized training sets
        (~2 500 samples); a maximum of 800 stratified samples is used.

This module provides ``_train_lightgbm``, ``_train_ebm``, and
``_train_rulefit`` as reusable building blocks.  The distillation and
RuleFit scripts import ``_train_ebm`` and ``_train_rulefit`` directly.
``train_fold()`` calls only ``_train_lightgbm`` and ``_compute_shap_values``.
"""

from typing import Dict, Any, List, Optional
import time
import warnings

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit

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

    In the standard walk-forward loop (``train_fold``), only LightGBM is
    trained per fold.  EBM and RuleFit are provided as helper methods used
    by the distillation and rule-extraction scripts.

    See module docstring for the full three-tier model architecture.

    Attributes:
        config (Dict[str, Any]):  Configuration dictionary.
        lgbm_config (Dict):       LightGBM hyperparameters from config.
        ebm_config (Dict):        EBM hyperparameters (used by distillation scripts).
        rulefit_config (Dict):    RuleFit hyperparameters (used by rule-extraction scripts).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise ModelTrainer from the project configuration.

        Args:
            config: Configuration dictionary.  Reads the 'models' section
                    for lightgbm, ebm, and rulefit hyperparameters.
        """
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
        """
        Convert triple-barrier labels from {−1, +1} to {0, 1}.

        LightGBM (objective='binary') and EBM expect labels in {0, 1}.
        The triple-barrier labeler produces {−1, +1} (label_binary).
        The mapping is: −1 → 0, +1 → 1.

        Args:
            y: Series with label_binary values (−1 or +1).

        Returns:
            Integer array with values in {0, 1}.
        """
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

        Hyperparameter rationale
        ~~~~~~~~~~~~~~~~~~~~~~~~
        The dataset has ~2 500 training samples per fold (3-year window) and
        11 features (15 for the regime variant).  Parameters are tuned for
        regularisation on a small tabular dataset:

        ``n_estimators=300``
            Balanced between underfitting (too few trees) and overfitting
            (too many).  No early stopping is used — the walk-forward
            validation set is the held-out evaluation set and must not
            influence training.

        ``num_leaves=20``
            Rule of thumb: num_leaves << 2^max_depth (2^6=64) to prevent
            overly complex trees on small data.  Limits variance.

        ``max_depth=6``
            Maximum tree depth.  Together with num_leaves=20, ensures
            shallow trees that generalise across regime changes.

        ``learning_rate=0.05``
            Standard moderate rate for boosted trees; works well with
            n_estimators=300.

        ``min_child_samples=20``
            Minimum observations per leaf (~0.8 % of fold size).  Prevents
            leaves based on very few observations.

        ``subsample=0.8, colsample_bytree=0.8``
            Stochastic gradient boosting: each tree sees 80 % of rows and
            80 % of features, reducing correlation between trees and
            improving generalisation (analogous to random forest bagging).

        ``reg_lambda=1.0``
            L2 regularisation on leaf weights.  Penalises large scores
            and improves generalisation across the walk-forward period.

        ``min_gain_to_split=0.01``
            Prunes splits that contribute negligible information gain;
            prevents the model from memorising noise.

        ``class_weight='balanced'``
            Accounts for the mild imbalance between take-profit (+1) and
            stop/time-barrier (−1) classes produced by the triple-barrier
            labeler with k=1.0.

        Args:
            X_train:     Training features (MultiIndex removed).
            y_train:     Training labels in {0, 1}.
            fold_number: Fold index for logging.

        Returns:
            Trained LGBMClassifier, or None if training fails.
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
                min_child_samples=self.lgbm_config.get('min_child_samples', 20),
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
        Train an Explainable Boosting Machine on one fold.

        This method is NOT called by ``train_fold``.  It is called by
        ``run_walk_forward_distillation.py`` for knowledge distillation:
        ``y_train`` in that context contains soft labels derived from
        LightGBM's probability outputs (temperature-scaled), and
        ``sample_weight`` encodes label confidence.

        EBM overview
        ~~~~~~~~~~~~
        EBM is a generalised additive model (GAM) with pairwise interactions,
        trained with gradient boosting.  Each feature contributes through a
        learned shape function ``f_j(x_j)``, making the model fully
        transparent: the contribution of every feature to every prediction
        can be read directly from the shape functions.

        Hyperparameter rationale
        ~~~~~~~~~~~~~~~~~~~~~~~~
        ``max_bins=128``
            With ~2 500 training samples, 256 bins yields ~10 observations
            per bin on average; 128 bins gives ~20, producing more stable
            shape function estimates with lower variance.

        ``max_interaction_bins=32``
            Bins for pairwise interaction terms.  Lower than max_bins because
            interaction space is sparse.

        ``interactions=10``
            Number of pairwise interaction terms to fit.  Enough to capture
            the most important feature interactions without overfitting.

        ``learning_rate=0.01``
            Slower learning rate than LightGBM; EBM uses many more boosting
            rounds (max_rounds=5000) to converge.

        ``max_rounds=5000``
            EBM cycles through all features in each round; 5 000 rounds
            with lr=0.01 provides sufficient capacity without overfitting.

        ``min_samples_leaf=10``
            Prevents shape function estimation from very few data points;
            ~0.4 % of fold size.

        Args:
            X_train:     Training features (MultiIndex removed).
            y_train:     Labels in {0, 1} — hard labels for direct training
                         or soft labels (0–1 floats) for distillation.
            fold_number: Fold index for logging.

        Returns:
            Trained ExplainableBoostingClassifier, or None if training fails.
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
        Train a RuleFit classifier on a stratified subsample.

        RuleFit overview
        ~~~~~~~~~~~~~~~~
        RuleFit (Friedman & Popescu 2008) generates IF-THEN rules by extracting
        decision paths from an ensemble of shallow trees, then fits a sparse
        linear model (Lasso) over the rule indicators plus original features.
        The result is a set of human-readable trading rules with explicit
        support and coefficient values.

        Scalability constraint
        ~~~~~~~~~~~~~~~~~~~~~~
        RuleFit's rule-generation step does not scale to fold-sized training
        sets.  Training on >800 samples becomes prohibitively slow with the
        ``imodels`` implementation.  A stratified subsample of 800 observations
        (preserving class proportions) is used instead.

        Feature name handling
        ~~~~~~~~~~~~~~~~~~~~~
        ``imodels`` RuleFit parses feature names when building rule strings.
        Special characters in names like ``macd_line`` or ``regime_prob_0``
        cause internal parsing errors.  Integer string indices ('0', '1', ...)
        are used during fitting; a ``feature_mapping`` dict is returned so
        that rule strings can be restored to human-readable names downstream.

        Knowledge distillation note
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``imodels`` RuleFitClassifier does not support ``sample_weight`` in
        ``fit()``.  When called from the distillation script, the soft-label
        signal is embedded directly in ``y_train`` (binarised soft probs)
        rather than via weights.

        Hyperparameter rationale
        ~~~~~~~~~~~~~~~~~~~~~~~~
        ``n_estimators=100``   — trees for rule generation; enough diversity.
        ``tree_size=3``        — max depth 3 = max 8 conditions per rule;
                                 keeps rules readable (≤3 conditions typical).
        ``max_rules=50``       — upper bound on rule count before L1 pruning.
        ``alpha=0.1``          — L1 strength; when set, imodels ignores
                                 max_rules and uses Lasso to auto-select
                                 the active rule set.
        ``include_linear=True``— adds linear terms alongside rule indicators;
                                 allows the model to capture smooth trends.

        Args:
            X_train:      Training features (MultiIndex removed).
            y_train:      Labels in {0, 1}.
            fold_number:  Fold index for logging.
            feature_names: Original feature names for the mapping dict.
            sample_weight: Ignored (imodels does not support it); present for
                           API consistency with the distillation caller.

        Returns:
            Tuple (RuleFitClassifier, feature_mapping) where feature_mapping
            maps integer string index → original feature name.
            Returns None if training fails.
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

            # sample_weight is not supported by imodels RuleFitClassifier;
            # the distillation signal is carried in y_train instead.
            if sample_weight is not None:
                logger.debug(
                    "  RuleFit: sample_weight provided but not supported "
                    "by imodels — ignored"
                )

            model.fit(X_arr, y_train, feature_names=simple_names)

            logger.success(
                f"  RuleFit fold {fold_number} trained  |  "
                f"{time.time() - start:.2f}s"
            )
            return model, feature_mapping

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
        Compute SHAP values for a trained LightGBM model.

        Uses ``shap.TreeExplainer``, which computes exact Shapley values by
        exploiting the tree structure (polynomial time vs. the exponential
        brute-force approach).  For binary classification LightGBM returns a
        list of two arrays [negative_class, positive_class]; only the positive
        class (take_profit, label=1) values are retained.

        SHAP values are accumulated across folds in the walk-forward loop to
        produce an out-of-sample feature importance ranking that is not
        contaminated by in-sample fitting.

        Args:
            model:        Trained LGBMClassifier.
            X_val_clean:  Validation features (MultiIndex removed).
            fold_number:  Fold index for logging.

        Returns:
            Tuple (shap_values, expected_value):
            - shap_values: ndarray of shape (n_val_samples, n_features).
            - expected_value: scalar baseline for the positive class.
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
        Train models for a single walk-forward fold.

        What is trained here
        ~~~~~~~~~~~~~~~~~~~~
        - **LightGBM** (Tier 1): full training set, hard labels.
        - **EBM**: SKIPPED — use ``run_walk_forward_distillation.py``.
        - **RuleFit**: SKIPPED — use ``run_rulefit_distillation.py``.
        - **SHAP**: computed on validation set for LightGBM.

        MultiIndex handling
        ~~~~~~~~~~~~~~~~~~~
        ``X_train`` and ``X_val`` from WalkForwardCV carry a (ticker, date)
        MultiIndex.  The MultiIndex is stripped for model training
        (``reset_index(drop=True)``), but ``X_val`` with the original MultiIndex
        is preserved in the returned dict so that ``backtest.py`` can align
        predictions with dates.

        Args:
            fold_data: Dict from ``WalkForwardCV.split()`` containing X_train,
                       y_train, X_val, y_val, feature_names, and date metadata.

        Returns:
            Dict with keys: fold_number, train_start, train_end, val_start,
            val_end, models, X_val (with MultiIndex), X_val_clean, y_val,
            y_val_binary, val_dates, feature_names, training_time,
            shap_values, shap_expected_value.
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
        """
        Run the full walk-forward training loop.

        Instantiates ``WalkForwardCV`` from config, iterates over all folds,
        and collects per-fold results.  The returned list is serialised to
        a ``.pkl`` file by the calling script (e.g. ``run_walk_forward.py``).

        Args:
            data: MultiIndex (ticker, date) DataFrame with engineered features
                  and label_binary (output of ``DataLoader.load_engineered_features()``).

        Returns:
            List of fold result dicts (one per fold), as returned by
            ``train_fold()``.
        """
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
