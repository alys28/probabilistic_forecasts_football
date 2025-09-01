import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def brier_loss(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)

def brier_objective(preds, train_data):
    """Custom Brier score objective function for LightGBM."""
    labels = train_data.get_label()
    # Transform raw predictions to probabilities using sigmoid
    probs = _sigmoid(preds)
    # Gradient: derivative of Brier score w.r.t. raw predictions
    grad = 2.0 * (probs - labels) * probs * (1.0 - probs)
    # Hessian: second derivative of Brier score w.r.t. raw predictions
    hess = 2.0 * probs * (1.0 - probs) * (1.0 - 2.0 * probs * (probs - labels))
    return grad, hess

def brier_eval(preds, train_data):
    """Custom Brier score evaluation metric for LightGBM."""
    labels = train_data.get_label()
    probs = _sigmoid(preds)
    brier = np.mean((probs - labels)**2)
    return 'brier', brier, False  # eval_name, eval_result, is_higher_better



class LightGBM:
    def __init__(self, use_calibration=True, **kwargs):
        # Balanced parameters for reasonable overfitting control
        self.params = {
            'boosting_type': 'gbdt',
            'objective': brier_objective,
            'metric': 'None',
            'num_leaves': 10,
            'max_depth': 4,
            'learning_rate': 0.015,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_data_in_leaf': 50,
            'min_gain_to_split': 0.5,
            'min_sum_hessian_in_leaf': 5.0,
            'max_bin': 255,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
        self.params.update(kwargs)
        
        self.num_boost_round = 2000
        self.model = None
        self.calibrator = None
        self.use_calibration = use_calibration
        self.feature_names = None

    def fit(self, X, y, val_X=None, val_y=None):
        # Standard validation split
        if val_X is None and val_y is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = val_X, val_y
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Reasonable early stopping
        callbacks = [
            log_evaluation(period=0),
            early_stopping(stopping_rounds=30, verbose=False)
        ]
        
        # Train with standard early stopping
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            feval=brier_eval,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        # Store feature info
        if hasattr(X, 'shape'):
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Calibration using validation data only
        if self.use_calibration:
            raw_preds_cal = self.model.predict(X_val, num_iteration=self.model.best_iteration)
            uncalibrated_probs_cal = _sigmoid(raw_preds_cal)
            
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(uncalibrated_probs_cal, y_val)

    def predict(self, X):
        """Return binary predictions (0 or 1)."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get probabilities and convert to binary predictions
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        """Return prediction probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get raw predictions and convert to probabilities
        raw_preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        uncalibrated_probs = _sigmoid(raw_preds)
        
        # Apply calibration if available
        if self.use_calibration and self.calibrator is not None:
            calibrated_probs = self.calibrator.predict(uncalibrated_probs)
            # Ensure probabilities are in valid range
            calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
            return np.column_stack([1 - calibrated_probs, calibrated_probs])
        else:
            # Return uncalibrated probabilities
            return np.column_stack([1 - uncalibrated_probs, uncalibrated_probs])

    def score(self, X, y):
        """Return accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)



def setup_xgboost_models(training_data, validation_data):
    models = {}
    timesteps = list(training_data.keys())
    
    for i, timestep in enumerate(timesteps):
        timestep = round(timestep, 3)
        
        # Prepare data
        X = training_data[timestep]
        y = np.array([row["label"] for row in X])
        X = np.array([row["rows"].reshape(-1) for row in X])
        y_val = np.array([row["label"] for row in validation_data[timestep]])
        X_val = np.array([row["rows"].reshape(-1) for row in validation_data[timestep]])
        
        # Train model with enhanced regularization
        model = LightGBM(use_calibration=False)
        model.fit(X, y, val_X=X_val, val_y=y_val)
        models[timestep] = model

        # Calculate training loss
        y_pred = model.predict_proba(X)[:, 1]  # Get probability predictions
        train_loss = brier_loss(y, y_pred)
        train_accuracy = model.score(X, y)

        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_loss = brier_loss(y_val, y_val_pred)  # Use Brier score to match objective function
        val_accuracy = model.score(X_val, y_val)
        print(f"Timestep {timestep:.2%}: Training Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")
        # Minimal progress indicator
        if (i + 1) % 50 == 0 or i == len(timesteps) - 1:
            print(f"Completed {i + 1}/{len(timesteps)} timesteps")
    
    return models