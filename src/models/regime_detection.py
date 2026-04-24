"""
Regime Detection Module

Detects market regimes using Gaussian Mixture Models (GMM).
Follows strict no-look-ahead principle: the GMM is retrained on each fold's
training data and applied in inference mode on validation data.

Regime information is added as additional features to the existing feature set,
allowing the classification models to condition their predictions on the current
market regime without training separate models per regime.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    Detects market regimes using a Gaussian Mixture Model (GMM).

    A GMM with three components is fitted on three features that capture
    the prevailing market environment: daily return, 20-day EWM volatility,
    and the VIX index. The three resulting regimes are ordered by their mean
    volatility level, so that regime 0 consistently corresponds to low-volatility
    conditions, regime 1 to intermediate conditions, and regime 2 to crisis or
    high-volatility periods.

    The GMM is fitted exclusively on training data at each walk-forward fold
    boundary. Regime labels and probabilities are then assigned to both the
    training and validation sets using the fitted model, ensuring no look-ahead
    bias is introduced.

    The output of this module — regime state and per-regime probabilities — is
    appended to the feature matrix as additional columns, allowing LightGBM and
    EBM to condition their predictions on the current market regime.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        n_regimes (int): Number of GMM components (default 3)
        model (GaussianMixture): Fitted GMM model
        scaler (StandardScaler): Scaler fitted on training regime features
        regime_order (Dict[int, int]): Mapping from GMM component index to
            volatility-ordered regime index
        regime_features (List[str]): Features used for regime detection
    """

    def __init__(self, config: Dict[str, Any], n_regimes: int = 3):
        """
        Initialize RegimeDetector.

        Args:
            config: Configuration dictionary
            n_regimes: Number of regimes (default 3: low, medium, high volatility)
        """
        self.config = config
        self.n_regimes = n_regimes

        self.model = None
        self.scaler = None
        self.regime_order = None

        self.regime_features = ['ret_1d', 'vol_20d', 'vix']

        logger.info(f"RegimeDetector initialized: GMM with {n_regimes} regimes")

    def _extract_regime_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract and clean the three features used for regime detection.

        Args:
            X: DataFrame with all features

        Returns:
            Array of shape (n_samples, 3) containing [ret_1d, vol_20d, vix]

        Raises:
            ValueError: If any of the required features are missing
        """
        missing = [f for f in self.regime_features if f not in X.columns]
        if missing:
            raise ValueError(f"Missing regime features: {missing}")

        features = X[self.regime_features].values

        if np.isnan(features).any():
            logger.warning("NaN values in regime features — applying forward fill")
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
        Build a mapping from GMM component indices to volatility-ordered regime indices.

        Regimes are ordered by their mean vol_20d value in the scaled feature space,
        so that regime 0 = lowest volatility and regime 2 = highest volatility.
        This ensures consistent regime labelling across folds regardless of the
        arbitrary component ordering produced by the GMM.

        Args:
            features_scaled: Scaled training features, shape (n_samples, 3)
            labels: Raw GMM component assignments, shape (n_samples,)

        Returns:
            Dictionary mapping original GMM index to ordered regime index
        """
        # vol_20d is the second feature (index 1) in the scaled array
        regime_vols = {}
        for regime in range(self.n_regimes):
            mask = labels == regime
            regime_vols[regime] = features_scaled[mask, 1].mean() if mask.sum() > 0 else 0.0

        sorted_regimes = sorted(regime_vols.items(), key=lambda x: x[1])
        regime_order = {
            original: ordered
            for ordered, (original, _) in enumerate(sorted_regimes)
        }

        logger.info(f"Regime ordering by volatility: {regime_order}")
        return regime_order

    def fit(self, X_train: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit the GMM regime detector on training data.

        The scaler is fitted on the training features and stored so that
        validation data can be transformed consistently. The regime ordering
        is also computed here and stored for use in predict().

        Args:
            X_train: Training features DataFrame

        Returns:
            Self for method chaining
        """
        try:
            logger.info("Fitting GMM regime detector...")

            features = self._extract_regime_features(X_train)

            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)

            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                n_init=10,
                max_iter=200
            )
            self.model.fit(features_scaled)
            labels = self.model.predict(features_scaled)

            self.regime_order = self._order_regimes_by_volatility(features_scaled, labels)

            # Compute and log regime statistics using ordered labels
            ordered_labels = np.array([self.regime_order[l] for l in labels])
            regime_stats = self.get_regime_stats(X_train, ordered_labels)
            logger.success("GMM regime detector fitted successfully")
            logger.info(f"\nRegime statistics:\n{regime_stats}")

            return self

        except Exception as e:
            logger.error(f"Error fitting regime detector: {str(e)}")
            self.model = None
            self.scaler = None
            self.regime_order = None
            return self

    def predict(
        self,
        X: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Predict regime states and probabilities for new data.

        Uses the scaler and GMM fitted on training data. Both regime labels
        and probabilities are reordered to match the volatility-based ordering
        established during fit().

        Args:
            X: Features DataFrame

        Returns:
            Tuple of (regime_state, regime_proba):
            - regime_state: array of integers in {0, 1, 2} ordered by volatility
            - regime_proba: array of shape (n_samples, n_regimes)
            Returns (None, None) if model is not fitted or prediction fails
        """
        if self.model is None or self.scaler is None:
            logger.warning("Regime detector not fitted, cannot predict")
            return None, None

        try:
            features = self._extract_regime_features(X)
            features_scaled = self.scaler.transform(features)

            labels = self.model.predict(features_scaled)
            proba = self.model.predict_proba(features_scaled)

            ordered_labels = np.array([self.regime_order[l] for l in labels])

            ordered_proba = np.zeros_like(proba)
            for original, ordered in self.regime_order.items():
                ordered_proba[:, ordered] = proba[:, original]

            return ordered_labels, ordered_proba

        except Exception as e:
            logger.error(f"Error predicting regimes: {str(e)}")
            return None, None

    def add_regime_features(
        self,
        X: pd.DataFrame,
        regime_state: np.ndarray,
        regime_proba: np.ndarray
    ) -> pd.DataFrame:
        """
        Append regime state and per-regime probabilities to the feature DataFrame.

        Adds four columns:
        - regime_state: integer in {0, 1, 2} (0 = low vol, 2 = high vol)
        - regime_prob_0: probability of low-volatility regime
        - regime_prob_1: probability of medium-volatility regime
        - regime_prob_2: probability of high-volatility regime

        Args:
            X: Original features DataFrame
            regime_state: Array of ordered regime states
            regime_proba: Array of shape (n_samples, n_regimes)

        Returns:
            DataFrame with regime columns appended
        """
        X_with_regime = X.copy()
        X_with_regime['regime_state'] = regime_state

        for i in range(self.n_regimes):
            X_with_regime[f'regime_prob_{i}'] = regime_proba[:, i]

        logger.info(f"Added {1 + self.n_regimes} regime features to DataFrame")
        return X_with_regime

    def get_regime_stats(
        self,
        X_train: pd.DataFrame,
        regime_state: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics for each regime.

        Expects regime_state to use the volatility-ordered labels (0, 1, 2)
        so that the statistics are interpretable and consistent across folds.

        Args:
            X_train: Training features DataFrame
            regime_state: Array of ordered regime states {0, 1, 2}

        Returns:
            DataFrame indexed by regime with columns:
            mean_return, std_return, mean_vix, count, pct
        """
        stats = []

        for regime in range(self.n_regimes):
            mask = regime_state == regime
            n_obs = mask.sum()

            if n_obs > 0:
                regime_data = X_train[mask]
                stat = {
                    'regime': regime,
                    'mean_return': regime_data['ret_1d'].mean(),
                    'std_return': regime_data['ret_1d'].std(),
                    'mean_vix': regime_data['vix'].mean(),
                    'count': int(n_obs),
                    'pct': n_obs / len(X_train) * 100
                }
            else:
                stat = {
                    'regime': regime,
                    'mean_return': 0.0,
                    'std_return': 0.0,
                    'mean_vix': 0.0,
                    'count': 0,
                    'pct': 0.0
                }

            stats.append(stat)

        return pd.DataFrame(stats).set_index('regime')

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit on training data and add regime features to both train and validation sets.

        This is the main method used in the walk-forward loop. The GMM is fitted
        exclusively on training data, then applied to both sets. This guarantees
        no look-ahead bias — the regime model has no access to validation data
        during fitting.

        Args:
            X_train: Training features DataFrame
            X_val: Validation features DataFrame

        Returns:
            Tuple of (X_train_with_regime, X_val_with_regime).
            Returns (X_train, X_val) unchanged if fitting or prediction fails.
        """
        self.fit(X_train)

        if self.model is None:
            logger.warning("Regime detection failed, returning original DataFrames")
            return X_train, X_val

        train_state, train_proba = self.predict(X_train)
        if train_state is None:
            logger.warning("Regime prediction failed on train, returning original DataFrames")
            return X_train, X_val

        X_train_with_regime = self.add_regime_features(X_train, train_state, train_proba)

        val_state, val_proba = self.predict(X_val)
        if val_state is None:
            logger.warning("Regime prediction failed on val, returning original DataFrames")
            return X_train, X_val

        X_val_with_regime = self.add_regime_features(X_val, val_state, val_proba)

        logger.success(
            f"Regime features added — train: {X_train_with_regime.shape}, "
            f"val: {X_val_with_regime.shape}"
        )

        return X_train_with_regime, X_val_with_regime