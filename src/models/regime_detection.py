"""
Regime Detection Module
========================

Classifies market regimes using a Gaussian Mixture Model (GMM) fitted on
three market-state descriptors: daily return, realized volatility, and VIX.

Role in the pipeline
--------------------
The RegimeDetector is called inside the walk-forward loop by
``run_walk_forward_regime.py`` and ``run_walk_forward_distillation.py``.
At each fold boundary it:

    1. Fits a GMM on the fold's **training** data only.
    2. Assigns regime labels and probabilities to the **training** set.
    3. Assigns regime labels and probabilities to the **validation** set
       using the already-fitted model (no re-fitting on validation data).
    4. Appends four columns to both sets:
           regime_state   — integer ∈ {0, 1, 2}, ordered by volatility
           regime_prob_0  — posterior probability of the low-vol regime
           regime_prob_1  — posterior probability of the medium-vol regime
           regime_prob_2  — posterior probability of the high-vol regime

These four columns enter the feature matrix X of LightGBM (and the
knowledge-distilled EBM and RuleFit derived from it), allowing the models
to condition their predictions on the prevailing market environment.

Why GMM?
---------
- **Soft assignment**: unlike k-means, GMM assigns a probability to each
  regime, not just a hard label.  The three ``regime_prob_*`` columns give
  the model a continuous, differentiable signal about regime uncertainty.
- **Elliptical clusters**: ``covariance_type='full'`` allows the GMM to
  model elongated, correlated clusters.  Market regimes have very different
  shapes in (ret_1d, vol_20d, vix) space: a crisis regime has high values on
  all three axes but also high variance, while a bull regime is compact.
- **Interpretability**: the three components map naturally to the three market
  states that practitioners recognise: Bull (low vol, positive return), Neutral
  (medium vol, mixed return), Bear/Crisis (high vol, negative return).

Why 3 components?
------------------
Three components is the minimum that separates the three qualitatively
distinct market environments.  Fewer components conflate Bull and Neutral;
more components produce unstable micro-regimes that are hard to interpret
and prone to fold-to-fold inconsistency.

Why these three features: (ret_1d, vol_20d, vix)?
---------------------------------------------------
- ``ret_1d``   — captures the sign of the current daily move; distinguishes
  strongly trending markets from range-bound ones within the same vol level.
- ``vol_20d``  — realized volatility (EWM span=20); measures the current
  volatility regime from price history alone.
- ``vix``      — implied volatility; the market's forward-looking fear gauge.
  High VIX can precede a realized volatility spike by days, giving the GMM an
  early-warning signal not captured by backward-looking vol_20d alone.

VIX is excluded from the model feature matrix X to avoid multicollinearity
with vol_20d and to keep the 11-feature set interpretable.  It is used here,
in the GMM, precisely because it adds orthogonal information to vol_20d.

Why StandardScaler?
--------------------
ret_1d (order of magnitude 1e-2), vol_20d (1e-2), and vix (10–80) are on
very different scales.  Without scaling, the GMM distance metric would be
dominated by vix, effectively ignoring the return and volatility signals.
StandardScaler is fitted on training data and applied to validation data
using the training statistics — consistent with the no-look-ahead principle.

No-look-ahead guarantee
------------------------
The GMM is re-fitted at every fold boundary using only the fold's training
window.  The scaler is also re-fitted at each fold.  Validation data is
processed with ``scaler.transform`` (not ``fit_transform``) and
``model.predict`` (not ``model.fit_predict``), ensuring zero information
from the validation period leaks into the regime assignments.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    Detects market regimes using a 3-component Gaussian Mixture Model.

    Fitted exclusively on fold training data; applied to validation data
    using the stored scaler and GMM (no look-ahead).  Regimes are
    volatility-ordered so that regime 0 = Bull (low vol), regime 1 =
    Neutral (medium vol), regime 2 = Bear/Crisis (high vol), consistently
    across all folds regardless of the GMM's arbitrary initialization.

    GMM inputs (3 features):
        ret_1d, vol_20d, vix
        (VIX is used here only; it is excluded from the model's feature
        matrix X to avoid multicollinearity with vol_20d.)

    GMM outputs appended to X:
        regime_state (int ∈ {0,1,2}), regime_prob_0, regime_prob_1,
        regime_prob_2

    Attributes:
        config (Dict[str, Any]):    Configuration dictionary.
        n_regimes (int):            Number of GMM components (default 3).
        model (GaussianMixture):    Fitted GMM; None before fit().
        scaler (StandardScaler):    Scaler fitted on training data; None before fit().
        regime_order (Dict[int,int]): GMM component → volatility-ordered index.
        regime_features (List[str]): ['ret_1d', 'vol_20d', 'vix'].
    """

    def __init__(self, config: Dict[str, Any], n_regimes: int = 3):
        """
        Initialise RegimeDetector.

        Args:
            config:    Configuration dictionary (not read directly; reserved
                       for future parameter overrides).
            n_regimes: Number of GMM components.  Default 3 = Bull / Neutral /
                       Bear.  See module docstring for rationale.
        """
        self.config = config
        self.n_regimes = n_regimes

        self.model        = None
        self.scaler       = None
        self.regime_order = None

        # The three market-state features fed to the GMM.
        # VIX is included here but excluded from the model feature matrix X.
        self.regime_features = ['ret_1d', 'vol_20d', 'vix']

        logger.info(
            f"RegimeDetector initialised  |  "
            f"GMM n_components={n_regimes}  |  "
            f"inputs: {self.regime_features}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_regime_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract the three GMM input features and handle NaN values.

        NaN handling
        ~~~~~~~~~~~~
        Forward fill propagates the last valid value; ``fillna(0)`` handles
        leading NaNs at the very start of the series (before vol_20d has
        warmed up, ~1 row).  In the walk-forward setting the training window
        starts in 2010, well after the warmup period, so leading NaN rows
        are essentially never present.  The fallback to 0 is a safe default
        that places any residual edge-case rows in the low-vol region of the
        scaled feature space.

        Args:
            X: DataFrame containing 'ret_1d', 'vol_20d', and 'vix' columns.

        Returns:
            NumPy array of shape (n_samples, 3).

        Raises:
            ValueError: If any of the required features are missing.
        """
        missing = [f for f in self.regime_features if f not in X.columns]
        if missing:
            raise ValueError(
                f"Required GMM features missing: {missing}.  "
                f"Available columns: {X.columns.tolist()}"
            )

        features = X[self.regime_features].values

        if np.isnan(features).any():
            logger.warning("NaN in GMM features — applying forward fill then fillna(0)")
            features = (
                pd.DataFrame(features, columns=self.regime_features)
                .ffill()
                .fillna(0)
                .values
            )

        return features

    def _order_regimes_by_volatility(
        self,
        features_scaled: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, int]:
        """
        Map GMM component indices to volatility-ordered regime indices.

        GMM component indices are arbitrary (depend on random initialization).
        Sorting by mean vol_20d in the scaled feature space gives a stable,
        interpretable ordering: 0 = lowest vol, 2 = highest vol, across all
        folds.

        vol_20d is index 1 in ``features_scaled`` (column order matches
        ``self.regime_features = ['ret_1d', 'vol_20d', 'vix']``).

        Args:
            features_scaled: Scaled training features, shape (n_samples, 3).
            labels:          Raw GMM component assignments, shape (n_samples,).

        Returns:
            Dict mapping original_component_index → ordered_regime_index.
        """
        regime_vols = {
            regime: (
                features_scaled[labels == regime, 1].mean()
                if (labels == regime).sum() > 0 else 0.0
            )
            for regime in range(self.n_regimes)
        }

        sorted_regimes = sorted(regime_vols.items(), key=lambda x: x[1])
        regime_order   = {
            original: ordered
            for ordered, (original, _) in enumerate(sorted_regimes)
        }

        logger.info(
            f"Regime ordering (GMM index → vol-ordered): {regime_order}  |  "
            f"mean vol_20d (scaled): "
            + "  ".join(f"r{k}={v:.3f}" for k, v in sorted(regime_vols.items()))
        )

        return regime_order

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit the GMM and StandardScaler on training data.

        GMM hyperparameters
        ~~~~~~~~~~~~~~~~~~~
        - ``covariance_type='full'``: each component has its own full
          covariance matrix, allowing elliptical clusters.  The three market
          regimes have very different shapes in feature space.
        - ``n_init=10``: the GMM is initialized 10 times with different random
          seeds; the best solution (highest log-likelihood) is kept.  This
          guards against convergence to a local optimum.
        - ``max_iter=200``: allows the EM algorithm to converge even in
          challenging cases (e.g. small training windows in early folds).
        - ``random_state=42``: reproducibility across runs.

        No-look-ahead guarantee
        ~~~~~~~~~~~~~~~~~~~~~~~
        ``scaler.fit_transform`` and ``model.fit`` are called on training data
        only.  The fitted objects are stored on ``self`` for later use in
        ``predict()``, which calls ``scaler.transform`` and ``model.predict``
        on new (validation) data without re-fitting.

        IMPORTANT: ``X_train`` must be the **full** fold DataFrame (including
        'vix', 'ret_1d'), not the 11-feature classification matrix.

        Args:
            X_train: Full training DataFrame with 'ret_1d', 'vol_20d', 'vix'.

        Returns:
            Self (for method chaining).
        """
        try:
            logger.info("Fitting GMM regime detector on training data...")

            features = self._extract_regime_features(X_train)

            self.scaler       = StandardScaler()
            features_scaled   = self.scaler.fit_transform(features)

            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                n_init=10,
                max_iter=200,
            )
            self.model.fit(features_scaled)
            labels = self.model.predict(features_scaled)

            self.regime_order  = self._order_regimes_by_volatility(features_scaled, labels)
            ordered_labels     = np.array([self.regime_order[l] for l in labels])

            regime_stats = self.get_regime_stats(X_train, ordered_labels)
            logger.success("GMM fitted successfully")
            logger.info(f"Regime statistics (training data):\n{regime_stats}")

            return self

        except Exception as e:
            logger.error(f"GMM fitting failed: {e}")
            self.model        = None
            self.scaler       = None
            self.regime_order = None
            return self

    def predict(
        self,
        X: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Assign regime labels and probabilities to new data.

        Uses the scaler and GMM stored from ``fit()``.  Both labels and
        probabilities are reordered to match the volatility-based ordering
        established during training.

        IMPORTANT: ``X`` must include 'vix', 'ret_1d', 'vol_20d' (full
        DataFrame, not the 11-feature classification matrix).

        Args:
            X: Full DataFrame with 'ret_1d', 'vol_20d', 'vix'.

        Returns:
            Tuple (regime_state, regime_proba):
            - regime_state: int array ∈ {0,1,2}, volatility-ordered.
            - regime_proba: float array, shape (n_samples, n_regimes),
              columns ordered by volatility (col 0 = low-vol probability).
            Returns (None, None) if the model is not fitted or prediction fails.
        """
        if self.model is None or self.scaler is None:
            logger.warning("RegimeDetector not fitted — cannot predict")
            return None, None

        try:
            features        = self._extract_regime_features(X)
            features_scaled = self.scaler.transform(features)   # transform, not fit_transform

            labels = self.model.predict(features_scaled)
            proba  = self.model.predict_proba(features_scaled)

            # Remap component indices to volatility-ordered regime indices
            ordered_labels = np.array([self.regime_order[l] for l in labels])

            ordered_proba = np.zeros_like(proba)
            for original, ordered in self.regime_order.items():
                ordered_proba[:, ordered] = proba[:, original]

            return ordered_labels, ordered_proba

        except Exception as e:
            logger.error(f"GMM prediction failed: {e}")
            return None, None

    def add_regime_features(
        self,
        X: pd.DataFrame,
        regime_state: np.ndarray,
        regime_proba: np.ndarray
    ) -> pd.DataFrame:
        """
        Append regime columns to the feature DataFrame.

        Adds four columns (the 4 regime features used by regime-aware models):
            regime_state   — int ∈ {0=Bull, 1=Neutral, 2=Bear/Crisis}
            regime_prob_0  — posterior P(low-vol regime)
            regime_prob_1  — posterior P(medium-vol regime)
            regime_prob_2  — posterior P(high-vol/crisis regime)

        Args:
            X:             Feature DataFrame to augment.
            regime_state:  Volatility-ordered regime labels, shape (n_samples,).
            regime_proba:  Ordered probabilities, shape (n_samples, n_regimes).

        Returns:
            Copy of X with four additional columns.
        """
        result = X.copy()
        result['regime_state'] = regime_state
        for i in range(self.n_regimes):
            result[f'regime_prob_{i}'] = regime_proba[:, i]

        logger.debug(
            f"Regime features appended: regime_state + "
            f"{[f'regime_prob_{i}' for i in range(self.n_regimes)]}"
        )
        return result

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit on training data and augment both train and validation sets.

        This is the primary method called inside the walk-forward loop.
        The GMM is fitted on ``X_train`` only, then applied to both sets —
        guaranteeing no validation data is seen during fitting.

        Fallback behaviour: if fitting or prediction fails for any reason,
        the original unmodified DataFrames are returned so that the
        walk-forward loop can continue without the regime features.

        Args:
            X_train: Full training DataFrame (with 'vix', 'ret_1d', 'vol_20d').
            X_val:   Full validation DataFrame (same columns required).

        Returns:
            Tuple (X_train_with_regime, X_val_with_regime), each augmented
            with regime_state and regime_prob_0/1/2.  Returns the originals
            unchanged if regime detection fails.
        """
        self.fit(X_train)

        if self.model is None:
            logger.warning("Regime fitting failed — returning original DataFrames")
            return X_train, X_val

        train_state, train_proba = self.predict(X_train)
        if train_state is None:
            logger.warning("Regime prediction failed on train — returning original DataFrames")
            return X_train, X_val

        X_train_regime = self.add_regime_features(X_train, train_state, train_proba)

        val_state, val_proba = self.predict(X_val)
        if val_state is None:
            logger.warning("Regime prediction failed on val — returning original DataFrames")
            return X_train, X_val

        X_val_regime = self.add_regime_features(X_val, val_state, val_proba)

        logger.success(
            f"Regime features added  |  "
            f"train: {X_train_regime.shape}  |  val: {X_val_regime.shape}"
        )

        return X_train_regime, X_val_regime

    # ------------------------------------------------------------------
    # Inspection utility
    # ------------------------------------------------------------------

    def get_regime_stats(
        self,
        X_train: pd.DataFrame,
        regime_state: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute per-regime descriptive statistics on training data.

        Useful for verifying that the volatility ordering is correct and
        that each regime has a meaningful interpretation:
            regime 0 → low mean_vix, positive mean_return  (Bull)
            regime 1 → intermediate                         (Neutral)
            regime 2 → high mean_vix, negative mean_return  (Bear/Crisis)

        Args:
            X_train:      Training DataFrame with 'ret_1d' and 'vix'.
            regime_state: Volatility-ordered regime labels, shape (n_samples,).

        Returns:
            DataFrame indexed by regime (0, 1, 2) with columns:
            mean_return, std_return, mean_vix, count, pct.
        """
        stats = []

        for regime in range(self.n_regimes):
            mask  = regime_state == regime
            n_obs = int(mask.sum())

            if n_obs > 0:
                rd = X_train[mask]
                stats.append({
                    'regime':      regime,
                    'mean_return': round(float(rd['ret_1d'].mean()), 5),
                    'std_return':  round(float(rd['ret_1d'].std()), 5),
                    'mean_vix':    round(float(rd['vix'].mean()), 2),
                    'count':       n_obs,
                    'pct':         round(n_obs / len(X_train) * 100, 1),
                })
            else:
                stats.append({
                    'regime': regime, 'mean_return': 0.0,
                    'std_return': 0.0, 'mean_vix': 0.0,
                    'count': 0, 'pct': 0.0,
                })

        return pd.DataFrame(stats).set_index('regime')
