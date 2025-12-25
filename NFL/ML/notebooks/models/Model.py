from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import optuna
import shap
import pandas as pd
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


class Model(ABC):
    """Abstract base class for ML models with shared utilities and HPO.

    Subclasses must implement the hyperparameter search space and the concrete
    training routine used by the optimizer.
    """

    def __init__(self, use_calibration: bool = False, optimize_hyperparams=False, numeric_features: Optional[List[str]] = None, other_features: Optional[List[str]] = None, all_features: Optional[List[str]] = None, n_trials=50) -> None:
        self.use_calibration = use_calibration
        self.best_params: Optional[Dict[str, Any]] = None
        self.model: Any = None
        self.calibrator = IsotonicRegression(out_of_bounds="clip") if use_calibration else None
        self.optimize_hyperparams = optimize_hyperparams
        # Preprocessing config
        self.numeric_features: Optional[List[str]] = numeric_features
        self.other_features: Optional[List[str]] = other_features
        self.all_features: Optional[List[str]] = all_features
        self.numeric_indices: Optional[List[int]] = None
        self.other_indices: Optional[List[int]] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.n_trials = n_trials
    # ---------- Shared metrics/utilities ----------
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def brier_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def split_data(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=False, # No shuffling because we dont want it to mess up the ordering of the matches (otherwise more data leakage than we would want)
            stratify=y if stratify else None,
        )

    # ---------- Calibration helpers ----------
    def fit_calibrator(self, uncalibrated_probs: np.ndarray, y_true: np.ndarray) -> None:
        self.calibrator.fit(uncalibrated_probs, y_true)

    def apply_calibration(self, probs: np.ndarray) -> np.ndarray:
        if self.use_calibration and self.calibrator is not None:
            calibrated = self.calibrator.predict(probs)
            return np.clip(calibrated, 0.0, 1.0)
        return probs

    # ---------- Abstracts required by subclasses ----------
    @abstractmethod
    def _define_search_space(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Return a dict of parameters sampled from the search space."""

    def _fixed_params(self) -> Dict[str, Any]:
        """Optional fixed params to merge into search params (subclass may override)."""
        return {}

    @abstractmethod
    def _train_model(self, x_train, y_train, x_val, y_val, params):
        pass

    # ---------- Preprocessing helpers (numeric scaling + passthrough) ----------
    def set_feature_config(self, numeric_features: List[str], other_features: List[str], all_features: List[str]) -> None:
        self.numeric_features = numeric_features
        self.other_features = other_features
        self.all_features = all_features
        self.numeric_indices = None
        self.other_indices = None
        self.preprocessor = None

    def _set_indices(self) -> None:
        if self.all_features is None:
            return
        if self.numeric_indices is None and self.numeric_features is not None:
            self.numeric_indices = [self.all_features.index(f) for f in self.numeric_features if f in self.all_features]
        if self.other_indices is None and self.other_features is not None:
            self.other_indices = [self.all_features.index(f) for f in self.other_features if f in self.all_features]

    def _build_preprocessor(self, n_columns: int) -> None:
        if self.all_features is None:
            self.preprocessor = None
            return
        self._set_indices()
        valid_numeric = [i for i in (self.numeric_indices or [])]
        valid_other = [i for i in (self.other_indices or [])]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), valid_numeric),
                ("other", "passthrough", valid_other),
            ],
            remainder="drop",
        )

    def fit_transform_X(self, X: np.ndarray) -> np.ndarray:
        if self.all_features is None:
            return X
        if self.preprocessor is None:
            self._build_preprocessor(X.shape[1])
        return self.preprocessor.fit_transform(X)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        if self.preprocessor is None or self.all_features is None:
            return X
        return self.preprocessor.transform(X)

    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        direction: str = "minimize",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run Optuna HPO using subclass-defined search space and training routine."""

        def objective(trial: optuna.trial.Trial) -> float:
            params = self._define_search_space(trial)
            params.update(self._fixed_params())
            val_metric = self._train_model(X_train, y_train, X_val, y_val, params)
            return float(val_metric)

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        return self.best_params

    # ---------- Common prediction API (optionally overridden) ----------
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1)."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get probabilities and convert to binary predictions
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)        

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities using fitted model. Subclasses should override for model-specific behavior."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        X_proc = self.transform_X(X)
        
        # Try to get predictions with best_iteration (for LightGBM) or without (for sklearn models)
        if hasattr(self.model, 'best_iteration'):
            raw_preds = self.model.predict(X_proc, num_iteration=self.model.best_iteration)
            uncalibrated_probs = self.sigmoid(raw_preds)
        else:
            # For sklearn models, get probabilities directly
            probs = self.model.predict_proba(X_proc)
            uncalibrated_probs = probs[:, 1]
        
        calibrated = self.apply_calibration(uncalibrated_probs)
        return np.column_stack([1 - calibrated, calibrated])

    def predict_proba_single(self, X):
        preds = self.predict_proba(X)
        return preds[:, 1]
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))
    
    def SHAP_analysis(self, X_test, X_train, plot = True):
        """
        Model interpretability with SHAP values
        """
        explainer = shap.Explainer(self.predict_proba_single, X_train, feature_names=self.all_features)
        shap_values = explainer(X_test)
        if plot:
            shap.plots.bar(shap_values)
        return shap_values
