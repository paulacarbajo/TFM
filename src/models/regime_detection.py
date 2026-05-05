"""
Regime Detection — 3-component GMM on (ret_1d, vol_20d, vix).

Called inside the walk-forward loop by run_walk_forward_regime.py and
run_walk_forward_distillation.py. At each fold:
    1. Fits GMM + StandardScaler on training data only.
    2. Assigns regime labels/probabilities to train and val via transform/predict.
    3. Appends four columns: regime_state (0=Bull, 1=Neutral, 2=Bear/Crisis),
       regime_prob_0, regime_prob_1, regime_prob_2.

GMM inputs use vol_20d (absolute volatility level) for regime clustering —
vol_rel is appropriate as a model feature but not here, since two periods
with different absolute vol but the same ratio would be wrongly conflated.
VIX adds a forward-looking signal absent from backward-looking vol_20d.
StandardScaler is required because vix (10–80) dwarfs ret_1d and vol_20d (1e-2).
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    3-component GMM regime detector. Fitted on training data only; applied
    to validation via transform/predict (no look-ahead).
    Regimes are volatility-ordered: 0=Bull, 1=Neutral, 2=Bear/Crisis.
    """

    def __init__(self, config: Dict[str, Any], n_regimes: int = 3):
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
        """Extract ['ret_1d', 'vol_20d', 'vix'] as NumPy array. Forward-fills NaNs, falls back to 0."""
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
        """Map arbitrary GMM component indices to volatility-ordered regime indices (0=low, 2=high).
        Orders by mean scaled vol_20d (index 1 in features_scaled)."""
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
        Fit StandardScaler and GMM on training data only.
        X_train must be the full fold DataFrame (with ret_1d, vol_20d, vix),
        not the 10-feature model matrix.
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
        Assign regime labels and probabilities using the fitted scaler and GMM.
        Returns (regime_state, regime_proba) or (None, None) if not fitted.
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
        """Append regime_state and regime_prob_0/1/2 columns to X.

        Note on SHAP: regime_state (integer 0/1/2) will always show SHAP≈0 in
        LightGBM because it is functionally redundant with regime_prob_0/1/2
        (regime_state = argmax(probs)). LightGBM finds no additional split gain
        from the discrete label when the continuous probabilities are available.
        This is expected behaviour, not a bug. regime_state is kept for human
        readability and downstream rule extraction in RuleFit.
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
        Primary method for the walk-forward loop. Fits on X_train, augments
        both train and val with regime columns. Returns originals if fitting fails.
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
        """Per-regime stats (mean_return, std_return, mean_vix, count, pct) for sanity checking."""
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
