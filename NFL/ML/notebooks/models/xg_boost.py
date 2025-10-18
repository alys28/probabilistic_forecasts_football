import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from .Model import Model


def brier_objective(preds, train_data):
    """Custom Brier score objective function for LightGBM."""
    labels = train_data.get_label()
    # Transform raw predictions to probabilities using sigmoid
    probs = Model.sigmoid(preds)
    # Gradient: derivative of Brier score w.r.t. raw predictions
    grad = 2.0 * (probs - labels) * probs * (1.0 - probs)
    # Hessian: second derivative of Brier score w.r.t. raw predictions
    hess = 2.0 * probs * (1.0 - probs) * (1.0 - 2.0 * probs * (probs - labels))
    return grad, hess

def brier_eval(preds, train_data):
    """Custom Brier score evaluation metric for LightGBM."""
    labels = train_data.get_label()
    probs = Model.sigmoid(preds)
    brier = np.mean((probs - labels)**2)
    return 'brier', brier, False  # eval_name, eval_result, is_higher_better



class LightGBM(Model):
    def __init__(self, use_calibration=True, optimize_hyperparams=False, n_trials=50, numeric_features=None, other_features=None, all_features=None, **kwargs):
        # Balanced parameters for reasonable overfitting control
        super().__init__(use_calibration=use_calibration, optimize_hyperparams=optimize_hyperparams, numeric_features=numeric_features, other_features=other_features, all_features=all_features)
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
        self.feature_names = None
        self.n_trials = n_trials
        self.best_params = None

    def _define_search_space(self, trial):
        """Define the hyperparameter search space for Bayesian optimization."""
        return {
            'num_leaves': trial.suggest_int('num_leaves', 5, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1.0, 20.0),
            'max_bin': trial.suggest_int('max_bin', 100, 500),
        }

    def _fixed_params(self):
        return {
            'boosting_type': 'gbdt',
            'objective': brier_objective,
            'metric': 'None',
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }
    def _train_model(self, X_train, y_train, X_val, y_val, params):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            log_evaluation(period=0),
            early_stopping(stopping_rounds=30, verbose=False)
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.num_boost_round,
            feval=brier_eval,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        raw_preds = model.predict(X_val, num_iteration=model.best_iteration)
        probs = self.sigmoid(raw_preds)
        return float(np.mean((probs - y_val) ** 2))

    def fit(self, X, y, val_X=None, val_y=None):
        # Standard validation split
        if val_X is None and val_y is None:
            X_train, X_val, y_train, y_val = self.split_data(X, y, test_size=0.25, random_state=42, stratify=True)
        else:
            X_train, y_train = X, y
            X_val, y_val = val_X, val_y
        
        # Preprocess features (scale numeric, passthrough others)
        X_train_proc = self.fit_transform_X(X_train)
        X_val_proc = self.transform_X(X_val)
        
        # Perform Bayesian optimization if requested
        if self.optimize_hyperparams:
            best = self.optimize_hyperparameters(X_train_proc, y_train, X_val_proc, y_val)
            self.params.update(best)
        
        # Create datasets
        train_data = lgb.Dataset(X_train_proc, label=y_train)
        val_data = lgb.Dataset(X_val_proc, label=y_val, reference=train_data)
        
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
        
        # Calibration using validation data only
        if self.use_calibration:
            raw_preds_cal = self.model.predict(X_val_proc, num_iteration=self.model.best_iteration)
            uncalibrated_probs_cal = self.sigmoid(raw_preds_cal)
            self.fit_calibrator(uncalibrated_probs_cal, y_val)

    def score(self, X, y):
        """Return accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)



def setup_xgboost_models(training_data, validation_data, numeric_features = None, other_features = None, use_calibration = True, optimize_hyperparams=False, n_trials=50):
    """
    Setup LightGBM models with optional Bayesian optimization.
    
    Args:
        training_data: Dictionary with timestep keys and training data
        validation_data: Dictionary with timestep keys and validation data
        optimize_hyperparams: Whether to perform Bayesian optimization
        n_trials: Number of optimization trials (only used if optimize_hyperparams=True)
    """
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
        
        # Train model with optional Bayesian optimization
        model = LightGBM(
            use_calibration=use_calibration, 
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials,
            numeric_features=numeric_features,
            other_features = other_features
        )
        model.fit(X, y, val_X=X_val, val_y=y_val)
        models[timestep] = model

        # Calculate training loss
        y_pred = model.predict_proba(X)[:, 1]  # Get probability predictions
        train_loss = Model.brier_loss(y, y_pred)
        train_accuracy = model.score(X, y)

        y_val_pred = model.predict_proba(X_val)[:, 1]
        val_loss = Model.brier_loss(y_val, y_val_pred)  # Use Brier score to match objective function
        val_accuracy = model.score(X_val, y_val)
        
        # Print results with optimization info
        opt_info = (f" (Optimized)" if optimize_hyperparams else "") + "(Calibrated)" if use_calibration else ""
        print(f"Timestep {timestep:.2%}{opt_info}: Training Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")
        
        # Minimal progress indicator
        if (i + 1) % 50 == 0 or i == len(timesteps) - 1:
            print(f"Completed {i + 1}/{len(timesteps)} timesteps")
    
    return models